"""Recuperacao densa: embeddings do Qwen3-Embedding-0.6B via HuggingFace.

    from src.embedding import QwenEmbedder

    embedder = QwenEmbedder()
    vetores = embedder.encode_documents(textos)        # (n, 1024), norma L2 = 1
    v_query = embedder.encode_queries([pergunta])[0]   # (1024,)

Captura similaridade semantica: parafrase, sinonimo, pergunta escrita com
palavras diferentes das do documento. Ver `src.esparso` para o lado lexical,
que cobre o que este perde.
"""

from .qwen import DEFAULT_MODEL, DEFAULT_TASK, QwenEmbedder

__all__ = ["DEFAULT_MODEL", "DEFAULT_TASK", "QwenEmbedder"]
