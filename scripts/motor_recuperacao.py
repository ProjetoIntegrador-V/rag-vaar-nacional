"""Motor de recuperação híbrida do VAAR (Tópico 5).

Mesma classe e mesmo método `buscar_legislacao()` da versão original; o que
mudou foi o necessário para rodar contra a coleção carregada pelo
notebooks/03_carga_qdrant.ipynb:

- imports pelos pacotes do repo (`src.embedding`, `src.esparso`), porque não
  existe módulo `qwen` nem `bm25` na raiz;
- conexão em nuvem (url + api_key) além da local por caminho;
- vetor esparso passa pelo vocabulário termo -> índice inteiro salvo na carga.
  O Qdrant só aceita índices inteiros, e o índice tem de ser o MESMO usado ao
  indexar, senão a busca lexical devolve zero;
- filtro de metadado (ano, tipo_documento) DENTRO de cada prefetch. Só no
  topo ele não propaga e vazam documentos de outros anos. Verificado;
- reranker opcional: pesa 2,2 GB e é lento em CPU;
- as etapas A/B/C/D ficaram em métodos separados para o chatbot cronometrar
  cada uma na aba Pipeline. `buscar_legislacao()` continua encadeando tudo;
- modo de busca selecionável (híbrida, densa, esparsa). Medido nesta máquina
  (4 núcleos, sem GPU): o Qdrant responde a consulta híbrida em 17 ms, e o
  Qwen leva 64 ms POR TOKEN para vetorizar a consulta. O custo é todo do
  embedding local, não do banco. O modo "esparsa" não chama o Qwen e responde
  em milissegundos, ao preço de perder a busca semântica.
"""
from __future__ import annotations

import os
import time
from collections import Counter
from typing import Any

from qdrant_client import QdrantClient, models

from src.embedding.qwen import DEFAULT_TASK, QwenEmbedder
from src.esparso import tokenize_pt

COLECAO_PADRAO = "vaar_rag"
RERANKER_PADRAO = "BAAI/bge-reranker-v2-m3"

# Teto de tokens da CONSULTA. Os chunks foram indexados com até 1.024 tokens,
# mas a consulta não precisa do mesmo teto: a 64 ms por token, cada token a
# mais custa caro. Medido com o parágrafo do HyDE, o vetor truncado em 256
# tokens tem cosseno 0,987 contra o vetor do texto inteiro (914 tokens), ou
# seja, aponta praticamente para a mesma direção por um terço do tempo.
MAX_TOKENS_CONSULTA = 256
# O reranker recebe o par (pergunta, chunk) e o chunk tem até 1.024 tokens;
# truncar aqui cortaria o documento, não a pergunta.
MAX_TOKENS_RERANK = 1024

MODOS_BUSCA = {
    "hibrida": "Híbrida (denso + esparso, RRF)",
    "esparsa": "Só esparsa (BM25, não carrega o Qwen)",
    "densa": "Só densa (Qwen)",
}


def usar_todos_os_nucleos() -> int:
    """O torch, por padrão, usa metade dos núcleos. Nesta máquina isso dobra o
    tempo do embedding à toa: 458 tokens caem de 53 s para 31 s com 4 threads."""
    import torch

    nucleos = os.cpu_count() or 1
    if torch.get_num_threads() < nucleos:
        torch.set_num_threads(nucleos)
    return torch.get_num_threads()


class MotorRecuperacaoVAAR:
    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        caminho_qdrant: str = "./banco_qdrant_local",
        colecao: str = COLECAO_PADRAO,
        vocabulario: dict[str, int] | None = None,
        motor_denso: QwenEmbedder | None = None,
        reranker: Any | None = None,
        carregar_reranker: bool = False,
        timeout: float = 60.0,
    ) -> None:
        # 1. Conexão: nuvem quando há url; senão local, que atende ao requisito
        #    de rodar offline e sem Docker.
        if url:
            self.banco = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        else:
            self.banco = QdrantClient(path=caminho_qdrant)
        self.colecao = colecao

        # 2. Qwen3-Embedding-0.6B, carregado só na primeira consulta quando não
        #    vier pronto: assim testar a conexão não baixa 1,2 GB.
        self._motor_denso = motor_denso

        # 3. Reranker Cross-Encoder. M3 é multilíngue e entende o vocabulário
        #    jurídico em português. Opcional porque pesa 2,2 GB.
        if reranker is None and carregar_reranker:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder(RERANKER_PADRAO, max_length=MAX_TOKENS_CONSULTA)
        self.reranker = reranker

        # 4. Vocabulário termo -> índice inteiro gerado na carga
        #    (data/vocabulario_esparso.json).
        self.vocabulario = vocabulario or {}

        # Cronômetro da última busca, lido pela aba Pipeline do chatbot para
        # mostrar quanto foi embedding e quanto foi banco.
        self.ultimo_tempo: dict[str, float] = {}
        self._ultimo_denso: tuple[str, list[float]] | None = None

    @property
    def motor_denso(self) -> QwenEmbedder:
        if self._motor_denso is None:
            usar_todos_os_nucleos()
            self._motor_denso = QwenEmbedder()
        # A trava de max_seq_length é obrigatória: sem sentence_bert_config.json
        # o sentence-transformers assume 32768 e estoura memória.
        self._motor_denso.model.max_seq_length = MAX_TOKENS_CONSULTA
        return self._motor_denso

    # ── ETAPA A: PREPARAÇÃO DOS VETORES ────────────────────────────────────
    def vetor_denso(self, texto: str) -> list[float]:
        # O vetor da consulta DEVE usar encode_queries() para aplicar o prefixo
        # de instrução; os documentos foram indexados sem prefixo (assimétrico).
        # Guarda o último: quando a busca filtrada volta vazia e o pipeline
        # repete sem filtro, seria absurdo pagar o embedding duas vezes.
        if self._ultimo_denso is not None and self._ultimo_denso[0] == texto:
            return self._ultimo_denso[1]
        vetor = self.motor_denso.encode_queries([texto], task=DEFAULT_TASK)[0].tolist()
        self._ultimo_denso = (texto, vetor)
        return vetor

    def vetor_esparso(self, texto: str) -> models.SparseVector | None:
        # Extrai as âncoras exatas (ex: "3106200", "art. 14"). Só entram termos
        # que já existem no vocabulário: termo novo não está no índice.
        frequencia_tokens = Counter(
            self.vocabulario[t] for t in tokenize_pt(texto) if t in self.vocabulario
        )
        if not frequencia_tokens:
            return None
        return models.SparseVector(
            indices=list(frequencia_tokens.keys()),
            values=[float(v) for v in frequencia_tokens.values()],
        )

    # ── ETAPA B: BUSCA HÍBRIDA NATIVA (Reciprocal Rank Fusion) ─────────────
    def buscar_hibrida(
        self,
        pergunta_original: str,
        texto_expandido_hyde: str,
        top_k_busca: int = 20,
        filtros: dict[str, Any] | None = None,
        modo: str = "hibrida",
    ) -> list[dict[str, Any]]:
        """Recebe a pergunta original (para o BM25) e a gerada pelo HyDE (para
        o Qwen). O Qdrant resolve a fusão RRF e aplica seu próprio IDF.

        `modo` escolhe quais lados entram. Em "esparsa" o Qwen nem é carregado,
        o que derruba a latência de dezenas de segundos para milissegundos numa
        máquina sem GPU, perdendo a busca semântica."""
        if modo not in MODOS_BUSCA:
            raise ValueError(f"modo de busca desconhecido: {modo}")
        condicoes = [
            models.FieldCondition(key=campo, match=models.MatchValue(value=valor))
            for campo, valor in (filtros or {}).items()
        ]
        filtro = models.Filter(must=condicoes) if condicoes else None

        self.ultimo_tempo = {}
        prefetch = []
        if modo in ("hibrida", "densa"):
            t = time.perf_counter()
            vetor = self.vetor_denso(texto_expandido_hyde)
            self.ultimo_tempo["denso_ms"] = (time.perf_counter() - t) * 1000
            prefetch.append(models.Prefetch(query=vetor, using="denso",
                                            limit=top_k_busca, filter=filtro))
        if modo in ("hibrida", "esparsa"):
            t = time.perf_counter()
            esparso = self.vetor_esparso(pergunta_original)
            self.ultimo_tempo["esparso_ms"] = (time.perf_counter() - t) * 1000
            if esparso is not None:
                prefetch.append(models.Prefetch(query=esparso, using="esparso",
                                                limit=top_k_busca, filter=filtro))
        if not prefetch:
            # modo esparso e nenhum termo da pergunta está no vocabulário
            return []

        t = time.perf_counter()
        resultados_brutos = self.banco.query_points(
            collection_name=self.colecao,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=filtro,
            limit=top_k_busca,
            with_payload=True,
        )
        self.ultimo_tempo["qdrant_ms"] = (time.perf_counter() - t) * 1000
        candidatos = []
        for hit in resultados_brutos.points:
            doc = dict(hit.payload)
            doc["score_rrf"] = float(hit.score)
            candidatos.append(doc)
        return candidatos

    # ── ETAPA C: CROSS-ENCODER RERANKING ───────────────────────────────────
    def rerankear(
        self, pergunta_original: str, candidatos: list[dict[str, Any]], top_k_final: int = 5
    ) -> list[dict[str, Any]]:
        """Avalia o par (pergunta original, chunk) para cada um dos Top 20 e
        reordena pela nota do cross-encoder. Sem reranker, mantém a ordem RRF."""
        if not candidatos:
            return []
        if self.reranker is None:
            return candidatos[:top_k_final]
        pares_para_avaliacao = [[pergunta_original, doc.get("texto", "")] for doc in candidatos]
        scores_rerank = self.reranker.predict(pares_para_avaliacao)
        for doc, score in zip(candidatos, scores_rerank):
            doc["score_rerank"] = float(score)
        return sorted(candidatos, key=lambda d: d["score_rerank"], reverse=True)[:top_k_final]

    # ── ETAPA D: HIERARQUIA DE CONTEXTO (Small-to-Big Retrieval) ───────────
    @staticmethod
    def estruturar_contexto(documentos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Mantém só o que a Parte 6 consome. `chunk_id` e `page` ficam porque
        o chatbot resolve o sub-chunk de volta à página inteira com eles."""
        return [
            {
                "titulo": doc.get("titulo"),
                "ano": doc.get("ano"),
                "tipo_documento": doc.get("tipo_documento"),
                "page": doc.get("page"),
                "chunk_id": doc.get("chunk_id"),
                "texto": doc.get("texto"),
                "score_rrf": doc.get("score_rrf"),
                "score_rerank": doc.get("score_rerank"),
            }
            for doc in documentos
        ]

    # ── encadeamento original ──────────────────────────────────────────────
    def buscar_legislacao(
        self,
        pergunta_original: str,
        texto_expandido_hyde: str,
        top_k_busca: int = 20,
        top_k_final: int = 5,
        filtros: dict[str, Any] | None = None,
        modo: str = "hibrida",
    ) -> list[dict[str, Any]]:
        """Executa a busca multicamadas exigida pelo Tópico 5: A -> B -> C -> D."""
        candidatos = self.buscar_hibrida(pergunta_original, texto_expandido_hyde,
                                         top_k_busca, filtros, modo)
        melhores = self.rerankear(pergunta_original, candidatos, top_k_final)
        return self.estruturar_contexto(melhores)
