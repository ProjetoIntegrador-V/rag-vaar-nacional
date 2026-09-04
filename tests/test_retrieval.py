"""Testes dos dois componentes de recuperacao.

Cobrem tokenizacao e BM25 sem baixar o modelo, para rodar em CI em segundos.
O Qwen tem um teste proprio marcado como `lento`, fora da rodada padrao.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embedding import QwenEmbedder
from src.esparso import BM25Retriever, tokenize_pt


# ── tokenizacao ───────────────────────────────────────────────────────────

def test_tokenize_remove_acento():
    assert tokenize_pt("Orçamento") == tokenize_pt("Orcamento")


def test_tokenize_preserva_ancoras_do_dominio():
    tokens = tokenize_pt("Portaria art.14 do codigo 3106200-1 em 2025/2026")
    assert "art.14" in tokens
    assert "3106200-1" in tokens
    assert "2025/2026" in tokens


def test_tokenize_remove_stopword_mas_preserva_numero():
    tokens = tokenize_pt("o municipio de 2026")
    assert "o" not in tokens
    assert "de" not in tokens
    assert "2026" in tokens
    assert "municipio" in tokens


def test_tokenize_preserva_negacao():
    # "nao" carrega significado em pergunta sobre habilitacao
    assert "nao" in tokenize_pt("municipios que nao foram habilitados")


# ── bm25 ──────────────────────────────────────────────────────────────────

TEXTOS = [
    "O VAAR e a parcela do FUNDEB condicionada a resultados educacionais.",
    "O ICMS Educacional distribui recursos aos municipios de Minas Gerais.",
    "A portaria art.14 define o codigo 3106200 para Belo Horizonte.",
]
IDS = ["c1", "c2", "c3"]


@pytest.fixture
def bm25():
    return BM25Retriever().fit(TEXTOS, IDS)


def test_bm25_acha_por_ancora_exata(bm25):
    assert bm25.score("codigo 3106200", top_k=3)[0][0] == "c3"


def test_bm25_acha_por_artigo_de_lei(bm25):
    assert bm25.score("art.14", top_k=3)[0][0] == "c3"


def test_bm25_ignora_query_sem_termo_em_comum(bm25):
    assert bm25.score("bicicleta astronomia", top_k=3) == []


def test_bm25_casa_sem_acento(bm25):
    assert bm25.score("municipios", top_k=3)[0][0] == "c2"


def test_bm25_ids_default_sao_posicionais():
    r = BM25Retriever().fit(TEXTOS)
    assert r.score("VAAR", top_k=1)[0][0] == "0"


def test_bm25_rejeita_ids_desalinhados():
    with pytest.raises(ValueError):
        BM25Retriever().fit(TEXTOS, ["so-um-id"])


def test_bm25_exige_fit_antes_de_score():
    with pytest.raises(RuntimeError):
        BM25Retriever().score("qualquer coisa")


def test_bm25_aceita_documento_vazio():
    """Documento que tokeniza para vazio nao pode quebrar o indice."""
    # Tres documentos, e nao dois, para o termo buscado ficar em menos da
    # metade do corpus e ter IDF positivo. Ver test_bm25_termo_muito_comum_zera_idf.
    r = BM25Retriever().fit(
        ["", "o VAAR do FUNDEB", "texto sem relacao alguma"],
        ["vazio", "ok", "outro"],
    )
    assert r.score("VAAR", top_k=3)[0][0] == "ok"


def test_bm25_termo_muito_comum_zera_idf():
    """Fixa o comportamento de IDF do BM25 Okapi para termo pouco discriminante.

    Com o termo presente em metade (ou mais) do corpus, o IDF vai a zero ou
    fica negativo e o corte default (`min_score=0.0`) devolve lista vazia.
    Nao e bug: e o Okapi dizendo que aquele termo nao separa nada. Este teste
    existe para que a equipe reconheca o sintoma em vez de achar que o indice
    quebrou, e para flagrar se algum dia trocarmos para BM25Plus.
    """
    r = BM25Retriever().fit(["fundeb aqui", "fundeb ali"], ["a", "b"])

    assert r.score("fundeb", top_k=2) == []
    assert len(r.score("fundeb", top_k=2, min_score=float("-inf"))) == 2


# ── qwen (lento: baixa ~1.2 GB do HuggingFace) ────────────────────────────

@pytest.mark.lento
def test_qwen_query_e_documento_geram_vetores_diferentes():
    """A assimetria query/documento tem que aparecer no vetor.

    Se este teste falhar com vetores identicos, o prefixo de instrucao nao
    esta sendo aplicado, e a recuperacao vai degradar em silencio.
    """
    import numpy as np

    embedder = QwenEmbedder()
    texto = "O VAAR e a parcela do FUNDEB condicionada a resultados."

    v_doc = embedder.encode_documents([texto])[0]
    v_query = embedder.encode_queries([texto])[0]

    assert v_doc.shape == (embedder.dimension,)
    assert np.isclose(np.linalg.norm(v_doc), 1.0, atol=1e-3), "vetor nao normalizado"
    assert not np.allclose(v_doc, v_query), "prefixo de instrucao nao foi aplicado"
