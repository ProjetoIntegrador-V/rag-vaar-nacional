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
        if "nota:" in prompt.lower():          # prompt de scripts/avaliacao_metricas.py
            return "Nota: 1.0"
        return ""


class RecuperadorFalso:
    """Mesma assinatura do Recuperador real. `docs_sem_filtro` permite simular
    o caso em que o filtro de metadado não casa com nada mas o corpus tem
    resposta."""

    def __init__(self, docs, docs_sem_filtro=None):
        self._docs = docs
        self._docs_sem_filtro = docs if docs_sem_filtro is None else docs_sem_filtro
        self.reranker = None
        self.filtros_recebidos = None
        self.chamadas_busca = []
        self.ultimo_tempo = {"denso_ms": 12.0, "qdrant_ms": 3.0}

    def buscar(self, texto_denso, texto_esparso, filtros=None, candidatos=20, modo="hibrida"):
        self.filtros_recebidos = filtros
        self.chamadas_busca.append({"filtros": filtros, "modo": modo})
        return list(self._docs if filtros else self._docs_sem_filtro)

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


# ── clientes de LLM ────────────────────────────────────────────────────────
def test_criar_cliente_rejeita_provedor_desconhecido():
    from src.pipeline import criar_cliente
    with pytest.raises(ValueError, match="desconhecido"):
        criar_cliente("provedor-x", "k", "m")


@pytest.mark.parametrize("provedor,chave,modelo,url,trecho", [
    ("anthropic", "", "claude-opus-5", None, "Anthropic"),
    ("groq", "", "llama-3.3-70b-versatile", None, "API key"),
    ("outro", "k", "m", "", "base_url"),
    ("groq", "k", "", None, "modelo"),
])
def test_criar_cliente_guardas(provedor, chave, modelo, url, trecho):
    from src.pipeline import criar_cliente
    with pytest.raises(ValueError, match=trecho):
        criar_cliente(provedor, chave, modelo, url)


def test_ollama_nao_exige_chave():
    from src.pipeline import ClienteOpenAICompativel, criar_cliente
    c = criar_cliente("ollama", "", "llama3.1")
    assert isinstance(c, ClienteOpenAICompativel)
    assert c.base_url.startswith("http://localhost:11434")


def test_catalogo_de_provedores_consistente():
    from src.pipeline import PROVEDORES
    for nome, info in PROVEDORES.items():
        if nome in ("anthropic", "outro"):
            continue
        assert info["base_url"].startswith("http"), nome
        assert info["modelo_sugerido"], nome


# ── motor de recuperação da equipe (sem rede, sem modelo) ─────────────────
def _motor_sem_conexao(vocab):
    from scripts.motor_recuperacao import MotorRecuperacaoVAAR
    m = MotorRecuperacaoVAAR.__new__(MotorRecuperacaoVAAR)
    m.vocabulario = vocab
    m.reranker = None
    m._motor_denso = None
    return m


def test_motor_vetor_esparso_usa_indices_inteiros_do_vocabulario():
    m = _motor_sem_conexao({"art": 7, "14": 3, "lei": 9})
    sv = m.vetor_esparso("art. 14 da Lei 14.113, art. 14")
    assert sv is not None
    assert set(sv.indices) == {7, 3, 9}, "termo fora do vocabulário (14.113) não pode entrar"
    assert dict(zip(sv.indices, sv.values))[7] == 2.0, "frequência conta repetições"
    assert all(isinstance(i, int) for i in sv.indices)


def test_motor_vetor_esparso_sem_termo_conhecido_devolve_none():
    m = _motor_sem_conexao({"fundeb": 1})
    assert m.vetor_esparso("receita de bolo") is None


def test_motor_rerankear_sem_reranker_mantem_ordem_rrf():
    m = _motor_sem_conexao({})
    docs = [{"texto": "a"}, {"texto": "b"}, {"texto": "c"}]
    assert m.rerankear("p", docs, top_k_final=2) == docs[:2]


def test_motor_rerankear_com_reranker_reordena_e_marca_score():
    class RerankerFalso:
        def predict(self, pares):
            return [0.1, 0.9, 0.5]
    m = _motor_sem_conexao({})
    m.reranker = RerankerFalso()
    docs = [{"texto": "a"}, {"texto": "b"}, {"texto": "c"}]
    saida = m.rerankear("p", docs, top_k_final=2)
    assert [d["texto"] for d in saida] == ["b", "c"]
    assert saida[0]["score_rerank"] == 0.9


def test_motor_estruturar_contexto_preserva_chunk_id_para_small_to_big():
    from scripts.motor_recuperacao import MotorRecuperacaoVAAR
    saida = MotorRecuperacaoVAAR.estruturar_contexto(
        [{"titulo": "T", "ano": 2026, "texto": "x", "chunk_id": "d_p1_c0_s2", "page": 1, "extra": 1}])
    assert saida[0]["chunk_id"] == "d_p1_c0_s2" and saida[0]["page"] == 1
    assert "extra" not in saida[0]


# ── modo de busca e queda do filtro ────────────────────────────────────────
def test_modo_de_busca_chega_ao_recuperador():
    rec = RecuperadorFalso(DOCS)
    Pipeline(LLMFalso(), rec, ConfigPipeline(modo_busca="esparsa")).executar("condicionalidades")
    assert rec.chamadas_busca[0]["modo"] == "esparsa"


def test_filtro_que_nao_casa_com_nada_e_repetido_sem_filtro():
    """O LLM pede ano=2026 + Portaria Interministerial, combinação que não
    existe na coleção. Em vez de responder 'sem contexto', repete sem filtro."""
    rec = RecuperadorFalso([], docs_sem_filtro=DOCS)
    t = Pipeline(LLMFalso(), rec).executar("quais condicionalidades habilitam ao VAAR em 2026")
    assert t.desfecho == DESFECHO_RESPONDIDA
    assert [c["filtros"] for c in rec.chamadas_busca] == [
        {"ano": 2026, "tipo_documento": "Resolução"}, None]
    busca = t.etapa("busca")
    assert busca.dados["filtro_removido"] == {"ano": 2026, "tipo_documento": "Resolução"}
    assert "ignorado" in busca.detalhe


def test_sem_resultado_nem_com_nem_sem_filtro_ainda_para():
    rec = RecuperadorFalso([], docs_sem_filtro=[])
    t = Pipeline(LLMFalso(), rec).executar("quais condicionalidades habilitam ao VAAR em 2026")
    assert t.desfecho == DESFECHO_SEM_CONTEXTO
    assert t.onde_parou() == "busca"


def test_busca_registra_onde_foi_o_tempo():
    t = Pipeline(LLMFalso(), RecuperadorFalso(DOCS)).executar("condicionalidades")
    assert t.etapa("busca").dados["tempo_interno"] == {"denso_ms": 12.0, "qdrant_ms": 3.0}
    assert "denso 12 ms" in t.etapa("busca").detalhe


def test_motor_modo_esparsa_nao_toca_no_qwen():
    """Garante o ganho de latência: em modo esparso o embedder nem é acessado."""
    from scripts.motor_recuperacao import MotorRecuperacaoVAAR

    class BancoFalso:
        def query_points(self, **kw):
            self.prefetch = kw["prefetch"]
            return type("R", (), {"points": []})()

    m = MotorRecuperacaoVAAR.__new__(MotorRecuperacaoVAAR)
    m.vocabulario = {"vaar": 1}
    m.colecao = "c"
    m.banco = BancoFalso()
    m.ultimo_tempo = {}
    m._ultimo_denso = None
    m._motor_denso = None          # tocar no Qwen aqui baixaria 1,2 GB

    m.buscar_hibrida("vaar", "documento hipotetico", 20, None, modo="esparsa")
    assert len(m.banco.prefetch) == 1
    assert m.banco.prefetch[0].using == "esparso"
    assert "denso_ms" not in m.ultimo_tempo


def test_motor_rejeita_modo_desconhecido():
    from scripts.motor_recuperacao import MotorRecuperacaoVAAR
    m = MotorRecuperacaoVAAR.__new__(MotorRecuperacaoVAAR)
    with pytest.raises(ValueError, match="modo de busca"):
        m.buscar_hibrida("a", "b", modo="magica")


# ── teto de contexto ───────────────────────────────────────────────────────
def test_contexto_curto_passa_intacto():
    docs = [{"texto": "a" * 100}, {"texto": "b" * 100}]
    saida, cortados = Pipeline._limitar_contexto(docs, 1000)
    assert saida == docs and cortados == 0


def test_teto_reparte_sobra_e_nao_corta_trecho_curto():
    """O trecho de 50 chars cabe inteiro mesmo ao lado de um gigante."""
    docs = [{"texto": "G" * 10_000}, {"texto": "p" * 50}]
    saida, cortados = Pipeline._limitar_contexto(docs, 1_000)
    assert cortados == 1
    assert saida[1]["texto"] == "p" * 50, "o curto nao pode ser cortado"
    assert saida[1].get("truncado") is None
    assert saida[0]["truncado"] is True
    assert saida[0]["texto"].endswith("[...]")
    assert sum(len(d["texto"]) for d in saida) <= 1_000 + len(" [...]")


def test_teto_respeita_o_orcamento_total():
    docs = [{"texto": "x" * 24_000} for _ in range(5)]
    saida, cortados = Pipeline._limitar_contexto(docs, 10_000)
    assert cortados == 5
    assert all(len(d["texto"]) <= 2_000 + len(" [...]") for d in saida)


def test_pipeline_corta_o_contexto_antes_de_gerar():
    llm = LLMFalso()
    docs = [{"titulo": "T", "ano": 2026, "page": 1, "chunk_id": "c0", "texto": "z" * 30_000}]
    t = Pipeline(llm, RecuperadorFalso(docs),
                 ConfigPipeline(limite_contexto_chars=2_000)).executar("condicionalidades")
    assert t.desfecho == DESFECHO_RESPONDIDA
    prompt_geracao = next(p for p in llm.chamadas if "EXCLUSIVAMENTE" in p)
    assert len(prompt_geracao) < 4_000, "o prompt tem de caber no orcamento"
    assert t.fontes[0]["truncado"] is True


def test_modo_esparso_nao_gasta_chamada_com_hyde():
    llm = LLMFalso()
    t = Pipeline(llm, RecuperadorFalso(DOCS),
                 ConfigPipeline(modo_busca="esparsa")).executar("condicionalidades")
    assert t.etapa("hyde").status == "pulado"
    assert "esparso" in t.etapa("hyde").detalhe
    assert not any("Nota Técnica do Inep" in p for p in llm.chamadas), "HyDE nao pode rodar"


# ── a resposta sai antes da avaliação ──────────────────────────────────────
def test_resposta_entregue_antes_da_avaliacao():
    """O juiz de factualidade é lento e não muda o texto: a interface recebe a
    resposta antes de ele rodar."""
    llm = LLMFalso()
    vistos = []

    def ao_responder(tr):
        vistos.append({"resposta": tr.resposta, "nota": tr.score_factualidade,
                       "etapas": [e.nome for e in tr.etapas]})

    t = Pipeline(llm, RecuperadorFalso(DOCS)).executar("condicionalidades",
                                                      ao_responder=ao_responder)
    assert len(vistos) == 1, "o callback roda uma vez"
    assert "Resolução CIF" in vistos[0]["resposta"], "a resposta ja estava pronta"
    assert vistos[0]["nota"] is None, "a nota ainda nao existia"
    assert "avaliacao" not in vistos[0]["etapas"], "a avaliacao ainda nao tinha rodado"
    assert t.score_factualidade == 1.0, "a nota chega depois"


def test_callback_nao_roda_quando_para_antes_da_geracao():
    vistos = []
    t = Pipeline(LLMFalso(aprovar=False), RecuperadorFalso(DOCS)).executar(
        "receita de bolo", ao_responder=lambda tr: vistos.append(tr))
    assert t.desfecho == DESFECHO_DESCARTADA
    assert vistos == []


def test_falha_ao_desenhar_nao_derruba_a_resposta():
    def quebrado(tr):
        raise RuntimeError("streamlit caiu")

    t = Pipeline(LLMFalso(), RecuperadorFalso(DOCS)).executar("condicionalidades",
                                                             ao_responder=quebrado)
    assert t.desfecho == DESFECHO_RESPONDIDA
    assert "Resolução CIF" in t.resposta
    assert "streamlit caiu" in t.etapa("geracao").dados["erro_ao_responder"]


# ── contagem de tokens ─────────────────────────────────────────────────────
class LLMComUso(LLMFalso):
    """Dublê que também reporta uso, como os clientes reais fazem."""

    def gerar_texto(self, prompt, **kw):
        r = super().gerar_texto(prompt, **kw)
        self.ultimo_uso = {"input_tokens": len(prompt) // 4, "output_tokens": len(r) // 4}
        return r


def test_tokens_sao_atribuidos_a_cada_etapa():
    llm = LLMComUso()
    t = Pipeline(llm, RecuperadorFalso(DOCS)).executar("quais condicionalidades do VAAR em 2026")
    assert t.desfecho == DESFECHO_RESPONDIDA
    com_llm = {"roteador", "reescrita", "metadados", "hyde", "geracao", "avaliacao"}
    for e in t.etapas:
        if e.nome in com_llm:
            assert e.tokens > 0, f"{e.nome} chamou o LLM e devia ter tokens"
        else:
            assert e.tokens == 0, f"{e.nome} nao chama LLM e devia ficar em zero"
    assert t.tokens_total() == sum(e.tokens for e in t.etapas)


def test_sem_uso_reportado_os_tokens_ficam_em_zero():
    """Provedor que não devolve `usage` não pode quebrar o pipeline."""
    t = Pipeline(LLMFalso(), RecuperadorFalso(DOCS)).executar("condicionalidades")
    assert t.desfecho == DESFECHO_RESPONDIDA
    assert t.tokens_total() == 0


@pytest.mark.parametrize("bruto,esperado", [
    ("on tokens per day (TPD): Limit 200000. Please try again in 2m19.968s.", "DIÁRIO"),
    ("on tokens per minute (TPM): Limit 8000. Please try again in 4.2s.", "POR MINUTO"),
    ("some other rate limit", "limite de requisições"),
])
def test_mensagem_do_429_distingue_dia_de_minuto(bruto, esperado):
    from src.pipeline.llm import _explicar_429
    msg = _explicar_429(bruto)
    assert esperado in msg
    assert ".." not in msg, "o tempo de espera nao pode sair com ponto duplicado"
