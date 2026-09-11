"""Orquestra os estágios e produz o Trace que a aba Pipeline exibe.

Fluxo:
    pergunta
      -> roteador            barra o que não é Fundeb/VAAR (para ANTES do banco)
      -> reescrita           leigo -> jargão normativo
      -> metadados           ano / tipo de documento -> filtro do Qdrant
      -> hyde                documento hipotético para o vetor denso
      -> busca               híbrida RRF, top-N candidatos (para se vier vazio)
      -> rerank              cross-encoder, top-k
      -> contexto_pai        sub-chunk -> página inteira
      -> geracao             resposta ancorada com citação
      -> avaliacao           factualidade, não bloqueia a resposta
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import etapas
from .llm import ClienteLLM
from .recuperacao import Recuperador
from .trace import (
    DESFECHO_DESCARTADA,
    DESFECHO_ERRO,
    DESFECHO_RESPONDIDA,
    DESFECHO_SEM_CONTEXTO,
    STATUS_OK,
    STATUS_PAROU,
    STATUS_PULADO,
    Trace,
)


@dataclass
class ConfigPipeline:
    usar_reescrita: bool = True
    usar_filtros: bool = True
    usar_hyde: bool = True
    usar_rerank: bool = True
    usar_contexto_pai: bool = True
    usar_avaliacao: bool = True
    candidatos: int = 20
    top_k: int = 5


class Pipeline:
    def __init__(self, llm: ClienteLLM, recuperador: Recuperador,
                 config: ConfigPipeline | None = None) -> None:
        self.llm = llm
        self.rec = recuperador
        self.cfg = config or ConfigPipeline()

    def executar(self, pergunta: str) -> Trace:
        t = Trace(pergunta=pergunta)
        try:
            self._executar(t)
        except Exception:  # noqa: BLE001
            # a etapa já foi marcada como erro dentro de `medir`
            t.desfecho = DESFECHO_ERRO
        return t

    # ── corpo ────────────────────────────────────────────────────────────
    def _executar(self, t: Trace) -> None:
        cfg = self.cfg

        # 1. roteador: filtro lógico antes do banco
        with t.medir("roteador") as e:
            rota = etapas.rotear(t.pergunta, self.llm)
            if rota.get("status") != "aprovado":
                e.status = STATUS_PAROU
                e.detalhe = rota.get("mensagem", "pergunta fora do escopo")
                t.resposta = e.detalhe
                t.desfecho = DESFECHO_DESCARTADA
                return
            e.detalhe = "aprovada: relacionada a Fundeb/VAAR"

        # 2. reescrita
        consulta = t.pergunta
        if cfg.usar_reescrita:
            with t.medir("reescrita") as e:
                consulta = etapas.reescrever(t.pergunta, self.llm)
                e.detalhe = consulta
                e.dados["consulta_reescrita"] = consulta
        else:
            t.registrar("reescrita", STATUS_PULADO, detalhe="desativada na configuração")

        # 3. filtros de metadado
        filtros: dict[str, Any] = {}
        if cfg.usar_filtros:
            with t.medir("metadados") as e:
                filtros = etapas.extrair_filtros(t.pergunta, self.llm)
                e.detalhe = ", ".join(f"{k}={v}" for k, v in filtros.items()) or "nenhum filtro explícito"
                e.dados["filtros"] = filtros
        else:
            t.registrar("metadados", STATUS_PULADO, detalhe="desativada na configuração")

        # 4. HyDE: o texto que vira vetor denso
        texto_denso = consulta
        if cfg.usar_hyde:
            with t.medir("hyde") as e:
                texto_denso = etapas.hyde(consulta, self.llm)
                e.detalhe = texto_denso[:220] + ("..." if len(texto_denso) > 220 else "")
                e.dados["documento_hipotetico"] = texto_denso
        else:
            t.registrar("hyde", STATUS_PULADO, detalhe="desativada na configuração")

        # 5. busca híbrida. O lado esparso recebe pergunta original + reescrita:
        #    as âncoras exatas ("art. 14") estão aí, não no documento inventado.
        texto_esparso = f"{t.pergunta} {consulta}" if consulta != t.pergunta else t.pergunta
        with t.medir("busca") as e:
            candidatos = self.rec.buscar(texto_denso, texto_esparso, filtros, cfg.candidatos)
            e.dados["n_candidatos"] = len(candidatos)
            e.dados["filtros_aplicados"] = filtros
            if not candidatos:
                e.status = STATUS_PAROU
                e.detalhe = "nenhum trecho retornou do banco vetorial"
                if filtros:
                    e.detalhe += f" com os filtros {filtros}"
                t.resposta = "A legislação recuperada não contém essa informação."
                t.desfecho = DESFECHO_SEM_CONTEXTO
                return
            e.detalhe = f"{len(candidatos)} candidatos" + (f" com filtro {filtros}" if filtros else "")
            e.dados["candidatos"] = [self._resumo_doc(d) for d in candidatos]

        # 6. reranking
        if cfg.usar_rerank and self.rec.reranker is not None:
            with t.medir("rerank") as e:
                top = self.rec.rerankear(t.pergunta, candidatos, cfg.top_k)
                e.detalhe = f"{len(candidatos)} -> {len(top)} pelo cross-encoder"
        else:
            top = candidatos[: cfg.top_k]
            motivo = "desativado na configuração" if not cfg.usar_rerank else "reranker não carregado"
            t.registrar("rerank", STATUS_PULADO,
                        detalhe=f"{motivo}; mantida a ordem do RRF, top {len(top)}")

        # 7. small-to-big
        if cfg.usar_contexto_pai:
            with t.medir("contexto_pai") as e:
                docs = self.rec.expandir_pais(top)
                expandidos = sum(1 for d in docs if d.get("expandido"))
                e.detalhe = f"{len(top)} sub-chunks -> {len(docs)} páginas ({expandidos} expandidas)"
        else:
            docs = top
            t.registrar("contexto_pai", STATUS_PULADO, detalhe="desativada na configuração")

        t.fontes = [self._resumo_doc(d) for d in docs]

        # 8. geração ancorada
        with t.medir("geracao") as e:
            t.resposta = etapas.gerar(t.pergunta, docs, self.llm)
            e.detalhe = f"{len(t.resposta)} caracteres, {len(docs)} fontes no contexto"
        t.desfecho = DESFECHO_RESPONDIDA

        # 9. avaliação: não bloqueia; erro aqui não derruba a resposta
        if cfg.usar_avaliacao:
            try:
                with t.medir("avaliacao") as e:
                    t.score_factualidade = etapas.avaliar(t.resposta, docs, self.llm)
                    e.detalhe = ("sem nota legível" if t.score_factualidade is None
                                 else f"factualidade = {t.score_factualidade:.1f}")
                    e.dados["score"] = t.score_factualidade
            except Exception:  # noqa: BLE001
                t.desfecho = DESFECHO_RESPONDIDA  # a resposta já existe
        else:
            t.registrar("avaliacao", STATUS_PULADO, detalhe="desativada na configuração")

    @staticmethod
    def _resumo_doc(d: dict[str, Any]) -> dict[str, Any]:
        return {
            "titulo": d.get("titulo"),
            "ano": d.get("ano"),
            "tipo_documento": d.get("tipo_documento"),
            "page": d.get("page"),
            "chunk_id": d.get("chunk_id"),
            "score_rrf": d.get("score_rrf"),
            "score_rerank": d.get("score_rerank"),
            "expandido": d.get("expandido", False),
            "trecho": (d.get("texto") or "")[:300],
            "texto": d.get("texto"),
        }
