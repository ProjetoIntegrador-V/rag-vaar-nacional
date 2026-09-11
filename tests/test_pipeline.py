"""Testes do pipeline com LLM e recuperador falsos. Não chamam rede nem modelo.

Cobrem os três desfechos que a aba Pipeline distingue:
  - barrada no roteador (antes do banco)
  - chegou ao banco, nada voltou (antes do LLM final)
  - respondida (passou por tudo)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import (  # noqa: E402
    DESFECHO_DESCARTADA,
    DESFECHO_RESPONDIDA,
    DESFECHO_SEM_CONTEXTO,
    ArmazemPais,
    ConfigPipeline,
    Pipeline,
)
from src.pipeline import etapas  # noqa: E402


# ── dublês ─────────────────────────────────────────────────────────────────
class LLMFalso:
    """Responde por palavra-chave do prompt, imitando o comportamento esperado."""

    def __init__(self, aprovar: bool = True):
        self.aprovar = aprovar
        self.chamadas: list[str] = []

    def gerar_texto(self, prompt, *, max_tokens=1024, effort="low", system=None):
        self.chamadas.append(prompt)
        if "classificador binário" in prompt:
            return "SIM" if self.aprovar else "NÃO"
        if "Reescrita:" in prompt:
            return "condicionalidades de habilitação à complementação-VAAR previstas no art. 14"
        if "SOMENTE com JSON" in prompt:
            return '{"ano": 2026, "tipo_documento": "Resolução"}'
        if "Nota Técnica do Inep" in prompt:
            return "Nos termos do art. 14 da Lei 14.113/2020, a habilitação ao VAAR..."
        if "EXCLUSIVAMENTE os trechos" in prompt:
            return "As condicionalidades são cinco. Fonte: Resolução CIF 24/2026."
        if "Nota:" in prompt:
            return "Nota: 1.0"
        return ""


class RecuperadorFalso:
    def __init__(self, docs):
        self._docs = docs
        self.reranker = None
        self.filtros_recebidos = None

    def buscar(self, texto_denso, texto_esparso, filtros=None, candidatos=20):
        self.filtros_recebidos = filtros
        return list(self._docs)

    def rerankear(self, pergunta, docs, top_k=5):
        return docs[:top_k]

    def expandir_pais(self, docs):
        return docs


DOCS = [
    {"titulo": "Resolução CIF nº 24/2026", "ano": 2026, "tipo_documento": "Resolução",
     "page": 3, "chunk_id": "res24_p3_c0_s1", "texto": "Art. 2º As condicionalidades..."},
    {"titulo": "Lei 14.113/2020", "ano": 2020, "tipo_documento": "Lei",
     "page": 9, "chunk_id": "lei14113_p9_c0", "texto": "Art. 14. A complementação-VAAR..."},
]


# ── os três desfechos ──────────────────────────────────────────────────────
def test_pergunta_fora_de_escopo_para_antes_do_banco():
    rec = RecuperadorFalso(DOCS)
    t = Pipeline(LLMFalso(aprovar=False), rec).executar("receita de bolo de cenoura")
    assert t.desfecho == DESFECHO_DESCARTADA
    assert t.onde_parou() == "roteador"
    assert not t.chegou_ao_banco()
    assert rec.filtros_recebidos is None, "o banco NAO pode ter sido consultado"
    assert t.etapa("busca") is None


def test_banco_vazio_para_antes_do_llm_final():
    llm = LLMFalso()
    t = Pipeline(llm, RecuperadorFalso([])).executar("quais condicionalidades do VAAR")
    assert t.desfecho == DESFECHO_SEM_CONTEXTO
    assert t.onde_parou() == "busca"
    assert t.chegou_ao_banco() is False        # busca existe mas status 'parou'
    assert t.etapa("geracao") is None
    assert not any("EXCLUSIVAMENTE" in p for p in llm.chamadas), "geracao nao pode rodar"


def test_fluxo_completo_respondida():
    llm = LLMFalso()
    rec = RecuperadorFalso(DOCS)
    t = Pipeline(llm, rec).executar("quais condicionalidades habilitam ao VAAR em 2026")
    assert t.desfecho == DESFECHO_RESPONDIDA
    assert t.onde_parou() == "nenhum"
    assert t.chegou_ao_banco() and t.chegou_ao_llm_final()
    assert t.score_factualidade == 1.0
    assert "Resolução CIF" in t.resposta
    assert rec.filtros_recebidos == {"ano": 2026, "tipo_documento": "Resolução"}
    nomes = [e.nome for e in t.etapas]
    assert nomes == ["roteador", "reescrita", "metadados", "hyde", "busca",
                     "rerank", "contexto_pai", "geracao", "avaliacao"]


def test_estagios_desativados_ficam_pulados_mas_visiveis():
    cfg = ConfigPipeline(usar_reescrita=False, usar_hyde=False, usar_avaliacao=False)
    t = Pipeline(LLMFalso(), RecuperadorFalso(DOCS), cfg).executar("condicionalidades")
    assert t.desfecho == DESFECHO_RESPONDIDA
    for nome in ("reescrita", "hyde", "avaliacao"):
        assert t.etapa(nome).status == "pulado"


def test_erro_no_llm_vira_desfecho_erro_sem_derrubar():
    class LLMQuebrado(LLMFalso):
        def gerar_texto(self, prompt, **kw):
            if "classificador" in prompt:
                return "SIM"
            raise RuntimeError("api fora do ar")

    t = Pipeline(LLMQuebrado(), RecuperadorFalso(DOCS)).executar("condicionalidades")
    assert t.desfecho == "erro"
    assert t.onde_parou() == "reescrita"
    assert "api fora do ar" in t.erro


# ── etapas isoladas ────────────────────────────────────────────────────────
def test_extrair_filtros_ignora_valor_inexistente():
    class L:
        def gerar_texto(self, p, **kw):
            return '```json\n{"ano": 1998, "tipo_documento": "Lei Complementar"}\n```'
    assert etapas.extrair_filtros("x", L()) == {}


def test_extrair_filtros_normaliza_caixa():
    class L:
        def gerar_texto(self, p, **kw):
            return '{"ano": "2026", "tipo_documento": "resolução"}'
    assert etapas.extrair_filtros("x", L()) == {"ano": 2026, "tipo_documento": "Resolução"}


@pytest.mark.parametrize("bruto,esperado", [
    ("1.0", 1.0), ("Nota: 0.0", 0.0), ("0,5", 0.5), ("A nota é 1", 1.0),
    ("7", 1.0), ("sem numero", None),
])
def test_avaliar_tolera_formatos(bruto, esperado):
    class L:
        def gerar_texto(self, p, **kw):
            return bruto
    assert etapas.avaliar("r", DOCS, L()) == esperado


def test_reescrita_cai_para_original_se_vier_lixo():
    class L:
        def gerar_texto(self, p, **kw):
            return ""
    assert etapas.reescrever("pergunta original", L()) == "pergunta original"


# ── chunk pai ──────────────────────────────────────────────────────────────
def test_armazem_pais_resolve_sufixo(tmp_path):
    jsonl = tmp_path / "c.jsonl"
    jsonl.write_text('{"chunk_id": "doc_p1_c0", "text": "PAGINA INTEIRA"}\n', encoding="utf-8")
    a = ArmazemPais(jsonl)
    assert a.id_pai("doc_p1_c0_s3") == "doc_p1_c0"
    assert a.id_pai("doc_p1_c0") == "doc_p1_c0"
    assert a.pai("doc_p1_c0_s3")["text"] == "PAGINA INTEIRA"
    assert a.pai("inexistente_s0") is None


def test_expandir_pais_deduplica_e_marca(tmp_path):
    from src.pipeline.recuperacao import Recuperador

    jsonl = tmp_path / "c.jsonl"
    jsonl.write_text('{"chunk_id": "doc_p1_c0", "text": "PAGINA INTEIRA"}\n', encoding="utf-8")
    rec = Recuperador.__new__(Recuperador)          # sem abrir conexao
    rec.pais = ArmazemPais(jsonl)
    docs = [{"chunk_id": "doc_p1_c0_s0", "texto": "pedaco 0"},
            {"chunk_id": "doc_p1_c0_s2", "texto": "pedaco 2"},
            {"chunk_id": "outro_p4_c0", "texto": "sem pai"}]
    saida = rec.expandir_pais(docs)
    assert len(saida) == 2, "dois sub-chunks da mesma pagina viram uma so"
    assert saida[0]["texto"] == "PAGINA INTEIRA" and saida[0]["expandido"] is True
    assert saida[1]["chunk_id"] == "outro_p4_c0" and "expandido" not in saida[1]


# ── guardas de credencial ──────────────────────────────────────────────────
@pytest.mark.parametrize("url,chave,trecho", [
    ("", "k", "endpoint"), ("   ", "k", "endpoint"), ("https://x", "", "API key"),
])
def test_recuperador_recusa_credencial_vazia(url, chave, trecho):
    from src.pipeline.recuperacao import Recuperador
    with pytest.raises(ValueError, match=trecho):
        Recuperador(qdrant_url=url, qdrant_api_key=chave, colecao="c",
                    embedder=None, vocabulario={})
