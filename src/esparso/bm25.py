"""Chamada do BM25 (retriever lexical), ajustada para portugues e para o dominio VAAR.

Componente isolado: recebe uma lista de textos, devolve scores por consulta.
Nao guarda nada em disco e nao depende de banco. A decisao de onde os textos e
os scores vao morar fica para quando o banco for definido.

Por que BM25 continua no pipeline mesmo com um modelo denso bom: embedding
denso "borra" ancoras de correspondencia exata. Numeros, siglas, codigos de
municipio, anos, numeros de portaria e artigos de lei sao exatamente o tipo de
token que o BM25 acerta e o denso erra. Nesse dominio isso e a regra, nao a
excecao ("VAAR 2026", "art. 14", "codigo IBGE 3106200").
"""

from __future__ import annotations

import re
import unicodedata
from typing import Sequence

# Stopwords em portugues, sem acento (o tokenizador remove acentos antes).
# Lista enxuta e proposital: nao removemos "nao", "sem", "mais" nem "menos",
# porque em pergunta sobre criterio de habilitacao elas carregam significado.
STOPWORDS_PT = {
    "a", "ao", "aos", "as", "da", "das", "de", "do", "dos", "e", "em", "na",
    "nas", "no", "nos", "o", "os", "ou", "para", "pela", "pelas", "pelo",
    "pelos", "por", "que", "se", "um", "uma", "umas", "uns", "com", "como",
    "sao", "foi", "ser", "sua", "seu", "suas", "seus", "isso", "este", "esta",
    "esse", "essa", "aquele", "aquela", "qual", "quais", "quando", "onde",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")


def strip_accents(text: str) -> str:
    """Remove acentos: 'orcamento' e 'orçamento' viram o mesmo token.

    Aplicado igualmente a documento e query. Isso importa porque usuario de
    portal publico digita sem acento com muita frequencia, e sem essa
    normalizacao o BM25 simplesmente nao casa.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def tokenize_pt(text: str) -> list[str]:
    """Tokenizador do indice lexical.

    Regras, e o motivo de cada uma:

    - minusculas + sem acento: normalizacao de superficie.
    - o padrao aceita '.', '-' e '/' NO MEIO do token, para nao picar
      "3106200-1", "art.14" ou "2025/2026" em pedacos inuteis.
    - stopwords caem fora, MENOS se forem numericas.
    - token de 1 caractere cai fora, menos digito.

    Nao usamos stemmer (RSLP). Stemmer aumenta recall em texto corrido, mas
    achata justamente as ancoras exatas que o BM25 esta aqui para preservar.
    Quem quer generalizacao morfologica ja tem o retriever denso ao lado.
    """
    text = strip_accents(text.lower())
    tokens = _TOKEN_RE.findall(text)
    return [
        t
        for t in tokens
        if (len(t) > 1 or t.isdigit()) and (t not in STOPWORDS_PT or t.isdigit())
    ]


class BM25Retriever:
    """BM25 Okapi em memoria sobre uma lista de textos.

    Uso:

        bm25 = BM25Retriever().fit(textos, ids)
        for doc_id, score in bm25.score("codigo 3106200", top_k=10):
            ...

    Escala: rank_bm25 e Python puro e varre o corpus inteiro por consulta.
    Serve confortavelmente ate a ordem de dezenas de milhares de documentos,
    que e a faixa deste projeto. Se passar disso, troque por `bm25s` (mesma
    ideia, ordens de grandeza mais rapido) ou por um indice invertido do
    proprio banco (OpenSearch, Postgres full-text). A assinatura de `score`
    nao muda, so o miolo.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25 = None
        self._ids: list[str] = []

    def fit(
        self, texts: Sequence[str], ids: Sequence[str] | None = None
    ) -> "BM25Retriever":
        """Constroi o indice. `ids` default e a posicao do texto na lista."""
        from rank_bm25 import BM25Okapi

        if ids is None:
            ids = [str(i) for i in range(len(texts))]
        if len(ids) != len(texts):
            raise ValueError(f"{len(ids)} ids para {len(texts)} textos")

        corpus = [tokenize_pt(t) for t in texts]
        # Documento que tokeniza para vazio (so simbolo, so stopword) quebraria
        # o calculo de tamanho medio de documento. Damos um token sentinela.
        corpus = [doc if doc else ["\x00vazio"] for doc in corpus]

        self._ids = list(ids)
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        return self

    def score(
        self, query: str, top_k: int = 10, min_score: float = 0.0
    ) -> list[tuple[str, float]]:
        """Devolve [(id, score_bm25)] ordenado por score decrescente.

        ATENCAO ao combinar com o denso mais tarde: o score do BM25 e
        ilimitado para cima e depende do tamanho do corpus, da raridade dos
        termos e do tamanho da query. Ele NAO e comparavel com similaridade de
        cosseno. Ver docs/retrieval-hibrida.md, secao sobre fusao.

        Sobre `min_score` (default 0.0, ou seja, so resultados estritamente
        positivos): o BM25 Okapi calcula

            idf(t) = log(N - df(t) + 0.5) - log(df(t) + 0.5)

        que vai a ZERO quando o termo aparece em metade dos documentos, e fica
        NEGATIVO acima disso. Ou seja, termo muito comum no corpus deixa de
        contribuir e pode ate empurrar o documento para baixo.

        Na pratica isso e desejavel: termo presente em quase todo documento nao
        discrimina nada. Mas num corpus pequeno e especializado como o do VAAR,
        palavras legitimas do dominio ("fundeb", "educacao", "municipio") podem
        cair nessa faixa e o lado lexical devolver lista vazia.

        Duas saidas, se isso acontecer:
        - passe `min_score=float("-inf")` para nao cortar nada; ou
        - troque `BM25Okapi` por `BM25Plus` no `fit()`, que usa
          idf = log((N + 1) / df), sempre positivo, e foi desenhado para
          corrigir exatamente esse comportamento.
        """
        if self._bm25 is None:
            raise RuntimeError("indice nao construido: chame fit() antes")

        tokens = tokenize_pt(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ordered = sorted(zip(self._ids, scores), key=lambda pair: pair[1], reverse=True)
        return [
            (doc_id, float(s)) for doc_id, s in ordered[:top_k] if s > min_score
        ]
