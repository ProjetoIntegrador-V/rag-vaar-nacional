"""Pipeline de recuperação e geração do RAG VAAR Nacional.

    from src.pipeline import Pipeline, ConfigPipeline, ClienteAnthropic, Recuperador

Os estágios que o LLM executa estão em `etapas.py` e reutilizam os scripts da
equipe em `scripts/`. A recuperação (Qdrant, reranking, chunk pai) está em
`recuperacao.py`. O `Trace` em `trace.py` registra por onde cada pergunta
passou e alimenta a aba Pipeline do chatbot.
"""

from .llm import MODELO_PADRAO, MODELOS, ClienteAnthropic, ClienteLLM, RespostaRecusada
from .orquestrador import ConfigPipeline, Pipeline
from .recuperacao import (
    ArmazemPais,
    Recuperador,
    carregar_embedder,
    carregar_reranker,
    carregar_vocabulario,
)
from .trace import (
    ANTES_DO_BANCO,
    DESFECHO_DESCARTADA,
    DESFECHO_ERRO,
    DESFECHO_RESPONDIDA,
    DESFECHO_SEM_CONTEXTO,
    ESTAGIOS,
    NO_BANCO,
    NO_LLM_FINAL,
    Etapa,
    Trace,
)

__all__ = [
    "ANTES_DO_BANCO", "NO_BANCO", "NO_LLM_FINAL", "ESTAGIOS",
    "DESFECHO_DESCARTADA", "DESFECHO_ERRO", "DESFECHO_RESPONDIDA", "DESFECHO_SEM_CONTEXTO",
    "ArmazemPais", "ClienteAnthropic", "ClienteLLM", "ConfigPipeline", "Etapa",
    "MODELOS", "MODELO_PADRAO", "Pipeline", "Recuperador", "RespostaRecusada", "Trace",
    "carregar_embedder", "carregar_reranker", "carregar_vocabulario",
]
