"""Camada de recuperação do chatbot: casca fina sobre o motor da equipe.

A busca híbrida, o reranking e a montagem do contexto vivem em
`scripts/motor_recuperacao.py` (`MotorRecuperacaoVAAR`). Esta classe só:

- valida credenciais antes de abrir conexão (url vazia faria o cliente
  apontar para localhost:6333 e o erro seria um "WinError 10061" opaco);
- expõe cada etapa do motor separadamente, para o orquestrador cronometrar
  busca e rerank como estágios distintos na aba Pipeline;
- acrescenta o small-to-big: troca o sub-chunk pela página original do JSONL,
  porque a carga reparticionou páginas em pedaços de 1.024 tokens (sufixo _sN).

Nada aqui chama LLM. Tudo é determinístico dado o texto de entrada.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scripts.motor_recuperacao import (
    MAX_TOKENS_CONSULTA,
    MAX_TOKENS_RERANK,
    MODOS_BUSCA,
    RERANKER_PADRAO,
    MotorRecuperacaoVAAR,
    usar_todos_os_nucleos,
)


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
        embedder: Any,
        vocabulario: dict[str, int],
        pais: ArmazemPais | None = None,
        reranker: Any | None = None,
        tarefa_embedding: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not (qdrant_url or "").strip():
            raise ValueError("endpoint do Qdrant ausente: preencha a URL do cluster")
        if not (qdrant_api_key or "").strip():
            raise ValueError("API key do Qdrant ausente")

        self.colecao = colecao
        self.pais = pais
        self.motor = MotorRecuperacaoVAAR(
            url=qdrant_url.strip(), api_key=qdrant_api_key.strip(), colecao=colecao,
            vocabulario=vocabulario, motor_denso=embedder, reranker=reranker, timeout=timeout,
        )

    @property
    def reranker(self):
        return self.motor.reranker

    @property
    def _cliente(self):
        return self.motor.banco

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

    # ── etapas do motor, expostas uma a uma ──────────────────────────────
    def vetor_denso(self, texto: str) -> list[float]:
        return self.motor.vetor_denso(texto)

    def vetor_esparso(self, texto: str):
        return self.motor.vetor_esparso(texto)

    def buscar(self, texto_denso: str, texto_esparso: str,
               filtros: dict[str, Any] | None = None, candidatos: int = 20,
               modo: str = "hibrida") -> list[dict[str, Any]]:
        """Etapas A e B do motor. `texto_denso` costuma ser o documento HyDE;
        `texto_esparso` a pergunta reescrita mais a original, porque as âncoras
        exatas ("art. 14") vivem na pergunta, não no documento inventado."""
        return self.motor.buscar_hibrida(texto_esparso, texto_denso, candidatos, filtros, modo)

    @property
    def ultimo_tempo(self) -> dict[str, float]:
        """Quanto da última busca foi embedding e quanto foi banco."""
        return self.motor.ultimo_tempo

    def rerankear(self, pergunta: str, docs: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        """Etapa C do motor."""
        return self.motor.rerankear(pergunta, docs, top_k)

    # ── small-to-big ─────────────────────────────────────────────────────
    def expandir_pais(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Etapa D: troca cada sub-chunk pela página original, sem repetir.

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
    import importlib.util

    # A instalação rápida (requirements-minimo.txt) não traz PyTorch nem
    # sentence-transformers. Sem esta checagem o usuário receberia um
    # ModuleNotFoundError cru ao trocar para a busca híbrida.
    if importlib.util.find_spec("sentence_transformers") is None:
        raise RuntimeError(
            "a busca semântica precisa do PyTorch e do sentence-transformers, "
            "que não estão instalados. Use o modo 'Só esparsa' na barra lateral, "
            "ou instale o restante com:  pip install -r requirements.txt"
        )

    from src.embedding import QwenEmbedder

    usar_todos_os_nucleos()
    emb = QwenEmbedder(model_name=modelo, batch_size=8)
    emb.model.max_seq_length = MAX_TOKENS_CONSULTA
    return emb


def carregar_reranker(modelo: str = RERANKER_PADRAO):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(modelo, max_length=MAX_TOKENS_RERANK)


def carregar_vocabulario(caminho: str | Path) -> dict[str, int]:
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} não existe. Ele é gerado pelo notebooks/03_carga_qdrant.ipynb "
            f"e é obrigatório: sem ele a busca esparsa devolve zero resultados."
        )
    return json.loads(caminho.read_text(encoding="utf-8"))
