"""Wrapper do Qwen3-Embedding-0.6B.

Tres detalhes do modelo que, se ignorados, degradam a busca sem gerar erro:

1. Pooling e de ULTIMO TOKEN (EOS), nao de media. O sentence-transformers ja
   configura isso pelo modelo. So importa se alguem reimplementar com
   transformers puro.

2. Query e documento sao codificados de forma ASSIMETRICA. A query leva um
   prefixo de instrucao, o documento nao. Por isso existem dois metodos aqui
   em vez de um `encode` generico. Codificar documento com prefixo de query
   (ou vice-versa) piora a recuperacao silenciosamente.

3. O tokenizer precisa de padding a esquerda, porque o pooling pega o ultimo
   token. Com padding a direita o vetor sai do token de padding.

Referencia: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Instrucao usada nas queries. O modelo foi treinado com esse formato:
#   "Instruct: {task}\nQuery:{query}"
# Trocar a instrucao muda os vetores de query, entao ela deve ficar fixa entre
# a indexacao e a consulta. Vale ajustar o texto para o dominio: o model card
# reporta ganho de 1 a 5 por cento com instrucao alinhada a tarefa.
DEFAULT_TASK = (
    "Dada uma pergunta sobre financiamento da educacao publica brasileira "
    "(FUNDEB, VAAR, ICMS Educacional), recupere os trechos de documentos "
    "que respondem a pergunta"
)


class QwenEmbedder:
    """Codifica textos em vetores densos normalizados (norma L2 = 1).

    Com os vetores normalizados, produto interno == similaridade de cosseno,
    o que deixa a busca densa ser uma unica multiplicacao de matriz.

    Parametros
    ----------
    model_name:
        Id no HuggingFace. Default vem de EMBEDDING_MODEL no .env.
    dim:
        Dimensao de saida. O modelo nativo entrega 1024 e suporta MRL
        (Matryoshka), entao da para truncar para 32..1024 sem retreinar.
        Truncar economiza armazenamento e acelera a busca, custando um pouco
        de qualidade. Mantenha 1024 enquanto o corpus for pequeno.
    device:
        "cuda", "cpu" ou None para autodetectar.
    """

    def __init__(
        self,
        model_name: str | None = None,
        dim: int | None = None,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        self.dim = dim
        self.batch_size = batch_size
        self._device = device
        self._model = None  # carregamento preguicoso: importar nao baixa 1.2 GB

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict = {"tokenizer_kwargs": {"padding_side": "left"}}
            if self.dim is not None:
                kwargs["truncate_dim"] = self.dim
            if self._device is not None:
                kwargs["device"] = self._device

            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    @property
    def dimension(self) -> int:
        return self.dim or self.model.get_sentence_embedding_dimension()

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Codifica chunks para indexacao. SEM prefixo de instrucao."""
        return self._encode(texts, prompt=None)

    def encode_queries(
        self, texts: Sequence[str], task: str = DEFAULT_TASK
    ) -> np.ndarray:
        """Codifica perguntas do usuario. COM prefixo de instrucao."""
        prompt = f"Instruct: {task}\nQuery:"
        return self._encode(texts, prompt=prompt)

    def _encode(self, texts: Sequence[str], prompt: str | None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors = self.model.encode(
            list(texts),
            prompt=prompt,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
        )
        return np.asarray(vectors, dtype=np.float32)
