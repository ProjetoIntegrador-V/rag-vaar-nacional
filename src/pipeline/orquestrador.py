"""Orquestra os estágios e produz o Trace que a aba Pipeline exibe.

Fluxo:
    pergunta
      -> roteador            barra o que não é Fundeb/VAAR (para ANTES do banco)
      -> reescrita           leigo -> jargão normativo
      -> metadados           ano / tipo de documento -> filtro do Qdrant
      -> hyde                documento hipotético para o vetor denso
      -> busca               híbrida RRF, top-N candidatos (para se vier vazio)
                         o modo (híbrida/densa/esparsa) é escolhido na interface
      -> rerank              cross-encoder, top-k
      -> contexto_pai        sub-chunk -> página inteira
      -> geracao             resposta ancorada com citação
      -> avaliacao           factualidade, não bloqueia a resposta

A avaliação é a etapa mais lenta depois da busca (o juiz relê todo o contexto)
e não altera uma vírgula do texto gerado. Por isso `executar` aceita
`ao_responder`: assim que a resposta existe, ela é entregue à interface, e só
então o juiz é chamado. Quem lê já está lendo enquanto a nota é calculada.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

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
    modo_busca: str = "hibrida"
    # Teto de caracteres do contexto enviado ao LLM. A expansão para o chunk
    # pai devolve a página inteira, e uma página de tabela da Portaria 14 tem
    # 24 mil caracteres: cinco delas estouram qualquer provedor. No plano
    # gratuito da Groq o limite é de 8.000 tokens por minuto, e o pipeline
    # gasta esse orçamento duas vezes (geração e avaliação).
    limite_contexto_chars: int = 10_000


class Pipeline:
    def __init__(self, llm: ClienteLLM, recuperador: Recuperador,
                 config: ConfigPipeline | None = None) -> None:
        self.llm = llm
        self.rec = recuperador
        self.cfg = config or ConfigPipeline()

    def executar(self, pergunta: str,
                 ao_responder: Callable[[Trace], None] | None = None) -> Trace:
        """`ao_responder` é chamado assim que a resposta fica pronta, antes da
        avaliação de factualidade. Serve para a interface mostrar o texto sem
        esperar o juiz. Não é chamado quando a pergunta para antes da geração."""
        t = Trace(pergunta=pergunta)
        try:
            self._executar(t, ao_responder)
        except Exception:  # noqa: BLE001
            # a etapa já foi marcada como erro dentro de `medir`
            t.desfecho = DESFECHO_ERRO
        return t

    # ── corpo ────────────────────────────────────────────────────────────
    def _executar(self, t: Trace, ao_responder: Callable[[Trace], None] | None = None) -> None:
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
        if cfg.modo_busca == "esparsa":
            # Sem lado denso não há o que vetorizar: gerar o parágrafo seria
            # uma chamada de LLM jogada fora.
            t.registrar("hyde", STATUS_PULADO,
                        detalhe="o modo esparso não usa vetor denso")
        elif cfg.usar_hyde:
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
            candidatos = self.rec.buscar(texto_denso, texto_esparso, filtros,
                                         cfg.candidatos, cfg.modo_busca)

            # O filtro de metadado é um palpite do LLM e às vezes pede uma
            # combinação que não existe: "Portaria 14 de 2026" vira
            # ano=2026 + Portaria Interministerial, mas a Portaria 14 está
            # indexada como 2025 (publicada em 2025, rege o exercício de 2026).
            # Antes de declarar "sem contexto", repete sem o filtro. O vetor
            # denso já está em cache no motor, então a repetição é de graça.
            filtro_removido = None
            if not candidatos and filtros:
                filtro_removido = dict(filtros)
                candidatos = self.rec.buscar(texto_denso, texto_esparso, None,
                                             cfg.candidatos, cfg.modo_busca)
                e.dados["filtro_removido"] = filtro_removido

            e.dados["n_candidatos"] = len(candidatos)
            e.dados["filtros_aplicados"] = {} if filtro_removido else filtros
            e.dados["modo_busca"] = cfg.modo_busca
            e.dados["tempo_interno"] = dict(getattr(self.rec, "ultimo_tempo", {}) or {})
            if not candidatos:
                e.status = STATUS_PAROU
                e.detalhe = "nenhum trecho retornou do banco vetorial"
                if filtros:
                    e.detalhe += f" nem com os filtros {filtros} nem sem eles"
                t.resposta = "A legislação recuperada não contém essa informação."
                t.desfecho = DESFECHO_SEM_CONTEXTO
                return
            e.detalhe = f"{len(candidatos)} candidatos ({cfg.modo_busca})"
            if filtro_removido:
                e.detalhe += f"; o filtro {filtro_removido} não casou com nada e foi ignorado"
            elif filtros:
                e.detalhe += f" com filtro {filtros}"
            tempos = e.dados["tempo_interno"]
            if tempos:
                e.detalhe += "; " + ", ".join(
                    f"{k.replace('_ms', '')} {v:.0f} ms" for k, v in tempos.items())
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
                chars = sum(len(d.get("texto") or "") for d in docs)
                if chars > cfg.limite_contexto_chars:
                    e.detalhe += (f"; {chars} caracteres cortados para "
                                  f"{cfg.limite_contexto_chars} (limite do provedor)")
        else:
            docs = top
            t.registrar("contexto_pai", STATUS_PULADO, detalhe="desativada na configuração")

        docs, cortados = self._limitar_contexto(docs, cfg.limite_contexto_chars)
        if cortados:
            t.etapa("contexto_pai") and t.etapa("contexto_pai").dados.update(
                {"trechos_cortados": cortados, "limite_chars": cfg.limite_contexto_chars})

        t.fontes = [self._resumo_doc(d) for d in docs]

        # 8. geração ancorada
        with t.medir("geracao") as e:
            t.resposta = etapas.gerar(t.pergunta, docs, self.llm)
            e.detalhe = f"{len(t.resposta)} caracteres, {len(docs)} fontes no contexto"
        t.desfecho = DESFECHO_RESPONDIDA

        # A resposta está pronta: entrega antes de chamar o juiz. Uma falha ao
        # desenhar fica registrada, mas não pode derrubar o que já foi gerado.
        if ao_responder is not None:
            try:
                ao_responder(t)
            except Exception as exc:  # noqa: BLE001
                t.etapa("geracao").dados["erro_ao_responder"] = f"{type(exc).__name__}: {exc}"

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
    def _limitar_contexto(docs: list[dict[str, Any]], limite: int) -> tuple[list[dict[str, Any]], int]:
        """Corta os textos para caber no orçamento, repartindo por igual.

        Reparte como uma torneira: quem já cabe na fatia leva o texto inteiro e
        devolve a sobra para os demais, então um trecho curto nunca é cortado
        por causa de um vizinho gigante. Devolve também quantos foram cortados.
        """
        if not docs or limite <= 0:
            return docs, 0
        total = sum(len(d.get("texto") or "") for d in docs)
        if total <= limite:
            return docs, 0

        sobra, restantes = limite, len(docs)
        cotas: dict[int, int] = {}
        for i, d in sorted(enumerate(docs), key=lambda par: len(par[1].get("texto") or "")):
            fatia = sobra // restantes
            usado = min(len(d.get("texto") or ""), fatia)
            cotas[i] = usado
            sobra -= usado
            restantes -= 1

        saida, cortados = [], 0
        for i, d in enumerate(docs):
            texto = d.get("texto") or ""
            if len(texto) <= cotas[i]:
                saida.append(d)
                continue
            novo = dict(d)
            novo["texto"] = texto[: cotas[i]].rstrip() + " [...]"
            novo["truncado"] = True
            saida.append(novo)
            cortados += 1
        return saida, cortados

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
            "truncado": d.get("truncado", False),
            "trecho": (d.get("texto") or "")[:300],
            "texto": d.get("texto"),
        }
