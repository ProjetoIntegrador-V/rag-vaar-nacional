"""Rastro de execução do pipeline.

Cada pergunta gera um `Trace` com uma `Etapa` por estágio percorrido. É isso
que a aba "Pipeline" do chatbot exibe: onde a pergunta passou, quanto tempo
levou em cada ponto e, se parou, em qual estágio e por quê.

Desfechos possíveis:
    respondida            passou por tudo e chegou ao LLM de geração
    descartada_intencao   o roteador barrou antes do banco vetorial
    sem_contexto          chegou ao banco, mas nada relevante voltou;
                          parou antes do LLM de geração
    erro                  exceção em algum estágio
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterator

STATUS_OK = "ok"
STATUS_PULADO = "pulado"
STATUS_PAROU = "parou"
STATUS_ERRO = "erro"

DESFECHO_RESPONDIDA = "respondida"
DESFECHO_DESCARTADA = "descartada_intencao"
DESFECHO_SEM_CONTEXTO = "sem_contexto"
DESFECHO_ERRO = "erro"

# Ordem canônica dos estágios. A aba Pipeline usa isto para mostrar também os
# estágios que NÃO foram alcançados, o que deixa claro onde a pergunta parou.
ESTAGIOS = [
    ("roteador", "Filtro de intenção"),
    ("reescrita", "Reescrita da consulta"),
    ("metadados", "Extração de filtros"),
    ("hyde", "Documento hipotético (HyDE)"),
    ("busca", "Busca híbrida no Qdrant"),
    ("rerank", "Reranking (cross-encoder)"),
    ("contexto_pai", "Expansão para o chunk pai"),
    ("geracao", "Geração ancorada"),
    ("avaliacao", "Avaliação de factualidade"),
]

# Fronteiras que a aba Pipeline destaca.
ANTES_DO_BANCO = {"roteador", "reescrita", "metadados", "hyde"}
NO_BANCO = {"busca", "rerank", "contexto_pai"}
NO_LLM_FINAL = {"geracao", "avaliacao"}


@dataclass
class Etapa:
    nome: str
    status: str
    duracao_ms: int = 0
    detalhe: str = ""
    dados: dict[str, Any] = field(default_factory=dict)
    # Tokens gastos nesta etapa. Só as etapas que chamam o LLM consomem;
    # busca, rerank e contexto pai ficam em zero.
    tokens_entrada: int = 0
    tokens_saida: int = 0

    @property
    def tokens(self) -> int:
        return self.tokens_entrada + self.tokens_saida

    @property
    def rotulo(self) -> str:
        return dict(ESTAGIOS).get(self.nome, self.nome)


@dataclass
class Trace:
    pergunta: str
    inicio: datetime = field(default_factory=datetime.now)
    etapas: list[Etapa] = field(default_factory=list)
    desfecho: str = DESFECHO_ERRO
    resposta: str | None = None
    fontes: list[dict[str, Any]] = field(default_factory=list)
    score_factualidade: float | None = None
    erro: str | None = None

    # ── consulta ─────────────────────────────────────────────────────────────
    def etapa(self, nome: str) -> Etapa | None:
        return next((e for e in self.etapas if e.nome == nome), None)

    def onde_parou(self) -> str:
        """Estágio em que a execução foi interrompida, ou 'nenhum'."""
        for e in self.etapas:
            if e.status in (STATUS_PAROU, STATUS_ERRO):
                return e.nome
        return "nenhum"

    def chegou_ao_banco(self) -> bool:
        return self.etapa("busca") is not None and self.etapa("busca").status == STATUS_OK

    def chegou_ao_llm_final(self) -> bool:
        return self.etapa("geracao") is not None and self.etapa("geracao").status == STATUS_OK

    def duracao_total_ms(self) -> int:
        return sum(e.duracao_ms for e in self.etapas)

    def resumo(self) -> str:
        """Frase curta para a lista da aba Pipeline."""
        if self.desfecho == DESFECHO_RESPONDIDA:
            return "Respondida: passou pelo banco e pelo LLM"
        if self.desfecho == DESFECHO_DESCARTADA:
            return "Barrada no filtro de intenção, antes do banco vetorial"
        if self.desfecho == DESFECHO_SEM_CONTEXTO:
            return "Chegou ao banco, nada relevante voltou; parou antes do LLM"
        return f"Erro em '{self.onde_parou()}'"

    # ── registro ─────────────────────────────────────────────────────────────
    def registrar(self, nome: str, status: str, duracao_ms: int = 0,
                  detalhe: str = "", **dados: Any) -> Etapa:
        e = Etapa(nome=nome, status=status, duracao_ms=duracao_ms,
                  detalhe=detalhe, dados=dados)
        self.etapas.append(e)
        return e

    @contextmanager
    def medir(self, nome: str, contador: Any = None) -> Iterator[Etapa]:
        """Cronometra um estágio. O corpo preenche `etapa.status` e `.detalhe`;
        se levantar exceção, o estágio é marcado como erro e a exceção sobe.

        `contador` é qualquer objeto com `total() -> (entrada, saida)`. A
        diferença entre antes e depois vira o custo em tokens da etapa, que a
        aba Pipeline exibe. Sem ele o campo fica zerado.
        """
        e = Etapa(nome=nome, status=STATUS_OK)
        self.etapas.append(e)
        antes = contador.total() if contador is not None else (0, 0)
        t0 = time.perf_counter()
        try:
            yield e
        except Exception as exc:  # noqa: BLE001
            e.status = STATUS_ERRO
            e.detalhe = f"{type(exc).__name__}: {exc}"
            self.erro = e.detalhe
            raise
        finally:
            e.duracao_ms = int((time.perf_counter() - t0) * 1000)
            if contador is not None:
                depois = contador.total()
                e.tokens_entrada = depois[0] - antes[0]
                e.tokens_saida = depois[1] - antes[1]

    def tokens_total(self) -> int:
        """Tokens gastos na pergunta inteira, entrada mais saída."""
        return sum(e.tokens for e in self.etapas)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["inicio"] = self.inicio.isoformat(timespec="seconds")
        d["onde_parou"] = self.onde_parou()
        d["duracao_total_ms"] = self.duracao_total_ms()
        return d
