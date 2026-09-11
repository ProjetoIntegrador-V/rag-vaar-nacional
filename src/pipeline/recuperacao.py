"""Camada de recuperação: Qdrant híbrido, reranking e expansão para o chunk pai.

Nada aqui chama LLM. Tudo é determinístico dado o texto de entrada, o que
permite testar a recuperação sem gastar token.

Pontos que não são óbvios e estão documentados no código:
- o filtro de metadado vai DENTRO de cada prefetch, senão vaza (ver buscar);
- o vetor esparso da consulta só usa termos que já existem no vocabulário;
- o reranker é opcional porque pesa mais de 2 GB;
- "chunk pai" aqui é a página original do JSONL, porque a carga reparticionou
  as páginas em pedaços de 1.024 tokens (sufixo _sN no chunk_id).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from src.esparso import tokenize_pt

NOME_DENSO = "denso"
NOME_ESPARSO = "esparso"
RERANKER_PADRAO = "BAAI/bge-reranker-v2-m3"
MAX_TOKENS_CONSULTA = 1024


class Embedder(Protocol):
    def encode_queries(self, texts: Sequence[str], task: str = ...) -> np.ndarray: ...


class Reranker(Protocol):
    def predict(self, pares: list[tuple[str, str]]) -> Sequence[float]: ...


# ── armazém de chunks pais (small-to-big) ─────────────────────────────────
class ArmazemPais:
    """Mapa chunk_id -> registro original do JSONL.

    A carga (notebooks/03) reparticiona páginas grandes em sub-chunks com
    sufixo `_s0`, `_s1`... Para devolver contexto inteiro ao LLM, este armazém
    resolve o sub-chunk de volta à página de origem.
    """

    _SUFIXO = re.compile(r"_s\d+$")

    def __init__(self, caminho_jsonl: str | Path) -> None:
        self._por_id: dict[str, dict[str, Any]] = {}
        caminho = Path(caminho_jsonl)
        if caminho.exists():
            with caminho.open(encoding="utf-8") as fh:
                for linha in fh:
                    linha = linha.strip()
                    if linha:
                        r = json.loads(linha)
                        self._por_id[r["chunk_id"]] = r

    def __len__(self) -> int:
        return len(self._por_id)

    def id_pai(self, chunk_id: str) -> str:
        return self._SUFIXO.sub("", chunk_id)

    def pai(self, chunk_id: str) -> dict[str, Any] | None:
        return self._por_id.get(self.id_pai(chunk_id))


# ── recuperador ────────────────────────────────────────────────────────────
class Recuperador:
    def __init__(
        self,
        *,
        qdrant_url: str,
        qdrant_api_key: str,
        colecao: str,
        embedder: Embedder,
        vocabulario: dict[str, int],
        pais: ArmazemPais | None = None,
        reranker: Reranker | None = None,
        tarefa_embedding: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        from qdrant_client import QdrantClient

        # Sem esta guarda, url vazia faz o cliente apontar para localhost:6333
        # e o usuário recebe um "WinError 10061" opaco em vez de saber que
        # esqueceu o endpoint.
        if not (qdrant_url or "").strip():
            raise ValueError("endpoint do Qdrant ausente: preencha a URL do cluster")
        if not (qdrant_api_key or "").strip():
            raise ValueError("API key do Qdrant ausente")

        self.colecao = colecao
        self.embedder = embedder
        self.vocabulario = vocabulario
        self.pais = pais
        self.reranker = reranker
        self.tarefa = tarefa_embedding
        self._cliente = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=timeout)

    # ── diagnóstico ──────────────────────────────────────────────────────
    def testar_conexao(self) -> tuple[bool, str]:
        try:
            if not self._cliente.collection_exists(self.colecao):
                nomes = [c.name for c in self._cliente.get_collections().collections]
                return False, (f"coleção '{self.colecao}' não existe; "
                               f"existentes: {nomes or 'nenhuma'}")
            n = self._cliente.count(self.colecao).count
            if n == 0:
                return False, f"coleção '{self.colecao}' existe mas está vazia"
            return True, f"coleção '{self.colecao}' com {n} pontos"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {str(exc)[:160]}"

    # ── vetores da consulta ──────────────────────────────────────────────
    def vetor_denso(self, texto: str) -> list[float]:
        if self.tarefa:
            v = self.embedder.encode_queries([texto], task=self.tarefa)[0]
        else:
            v = self.embedder.encode_queries([texto])[0]
        return np.asarray(v, dtype=np.float32).tolist()

    def vetor_esparso(self, texto: str):
        """Só termos já presentes no vocabulário da coleção. Termo novo não
        existe no índice e geraria um índice órfão."""
        from qdrant_client import models

        contagem: dict[int, float] = {}
        for tok in tokenize_pt(texto):
            idx = self.vocabulario.get(tok)
            if idx is not None:
                contagem[idx] = contagem.get(idx, 0.0) + 1.0
        if not contagem:
            return None
        return models.SparseVector(indices=list(contagem), values=list(contagem.values()))

    # ── busca híbrida ────────────────────────────────────────────────────
    def buscar(self, texto_denso: str, texto_esparso: str,
               filtros: dict[str, Any] | None = None, candidatos: int = 20) -> list[dict[str, Any]]:
        """Denso + esparso fundidos por RRF nativo do Qdrant.

        `texto_denso` costuma ser o documento hipotético (HyDE); `texto_esparso`
        a pergunta reescrita mais a original, porque as âncoras exatas
        ("art. 14") vivem na pergunta, não no documento inventado.
        """
        from qdrant_client import models

        condicoes = []
        for campo, valor in (filtros or {}).items():
            condicoes.append(models.FieldCondition(key=campo, match=models.MatchValue(value=valor)))
        filtro = models.Filter(must=condicoes) if condicoes else None

        # O filtro tem de ir DENTRO de cada prefetch. Só no topo ele não
        # propaga: cada lado traz candidatos sem filtro e o RRF apenas
        # reordena, então vazam documentos de outros anos. Verificado.
        prefetch = [models.Prefetch(query=self.vetor_denso(texto_denso),
                                    using=NOME_DENSO, limit=candidatos, filter=filtro)]
        esparso = self.vetor_esparso(texto_esparso)
        if esparso is not None:
            prefetch.append(models.Prefetch(query=esparso, using=NOME_ESPARSO,
                                            limit=candidatos, filter=filtro))

        resultado = self._cliente.query_points(
            collection_name=self.colecao,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=filtro,
            limit=candidatos,
            with_payload=True,
        )
        docs = []
        for p in resultado.points:
            d = dict(p.payload)
            d["score_rrf"] = float(p.score)
            docs.append(d)
        return docs

    # ── reranking ────────────────────────────────────────────────────────
    def rerankear(self, pergunta: str, docs: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not docs:
            return []
        if self.reranker is None:
            return docs[:top_k]
        pares = [(pergunta, d.get("texto", "")) for d in docs]
        scores = self.reranker.predict(pares)
        for d, s in zip(docs, scores):
            d["score_rerank"] = float(s)
        return sorted(docs, key=lambda d: d["score_rerank"], reverse=True)[:top_k]

    # ── small-to-big ─────────────────────────────────────────────────────
    def expandir_pais(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Troca cada sub-chunk pela página original, sem repetir a mesma página.

        Mantém a ordem do reranking: o primeiro pai a aparecer é o do melhor
        sub-chunk. Quando o pai não existe no armazém, o próprio chunk segue.
        """
        if self.pais is None or len(self.pais) == 0:
            return docs
        vistos: set[str] = set()
        saida = []
        for d in docs:
            cid = d.get("chunk_id", "")
            pid = self.pais.id_pai(cid)
            if pid in vistos:
                continue
            vistos.add(pid)
            pai = self.pais.pai(cid)
            if pai is None:
                saida.append(d)
                continue
            novo = dict(d)
            novo["texto"] = pai["text"]
            novo["chunk_id"] = pid
            novo["expandido"] = cid != pid
            saida.append(novo)
        return saida


# ── fábricas dos modelos pesados ───────────────────────────────────────────
def carregar_embedder(modelo: str = "Qwen/Qwen3-Embedding-0.6B"):
    """Qwen3 com a trava de tokens. Sem sentence_bert_config.json o
    sentence-transformers assume 32768 e estoura memória em texto longo."""
    from src.embedding import QwenEmbedder

    emb = QwenEmbedder(model_name=modelo, batch_size=8)
    emb.model.max_seq_length = MAX_TOKENS_CONSULTA
    return emb


def carregar_reranker(modelo: str = RERANKER_PADRAO):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(modelo, max_length=1024)


def carregar_vocabulario(caminho: str | Path) -> dict[str, int]:
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} não existe. Ele é gerado pelo notebooks/03_carga_qdrant.ipynb "
            f"e é obrigatório: sem ele a busca esparsa devolve zero resultados."
        )
    return json.loads(caminho.read_text(encoding="utf-8"))
