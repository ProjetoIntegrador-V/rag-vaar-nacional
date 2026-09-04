"""Recuperacao esparsa (lexical): BM25 Okapi com tokenizacao pt-BR.

    from src.esparso import BM25Retriever

    bm25 = BM25Retriever().fit(textos, ids)
    resultados = bm25.score("codigo 3106200", top_k=10)   # [(id, score)]

Captura ancoras de correspondencia exata: codigos de municipio, numeros de
portaria, artigos de lei, anos, siglas. E justamente o que o vetor denso borra.
Ver `src.embedding` para o lado semantico, que cobre o que este perde.
"""

from .bm25 import STOPWORDS_PT, BM25Retriever, strip_accents, tokenize_pt

__all__ = ["BM25Retriever", "STOPWORDS_PT", "strip_accents", "tokenize_pt"]
