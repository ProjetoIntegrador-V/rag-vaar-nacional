"""Estágios do pipeline que dependem do LLM.

Os quatro estágios que a equipe já escreveu são importados de `scripts/` e
usados como estão: roteador de intenção, HyDE, geração ancorada e avaliação
de factualidade. A avaliação recebe o LLM embrulhado num proxy que reduz a
saída ao primeiro número, porque a função original faz `float(texto)` direto
e quebraria se o modelo escrevesse "Nota: 1.0" ou "0,5".

Dois estágios são novos, previstos na arquitetura alvo (docs/arquitetura-alvo.md)
e ainda não existiam no código: reescrita da consulta e extração de filtros.
"""
from __future__ import annotations

import json
import re
from typing import Any

from scripts.avaliacao_metricas import avaliar_factualidade
from scripts.filtro_intencao import roteador_de_intencao
from scripts.geracao_hyde import gerar_documento_hyde
from scripts.geracao_rag_final import gerar_resposta_final

from .llm import ClienteLLM

# Valores que existem de fato nos metadados da coleção. Passar a lista ao
# modelo evita que ele invente um tipo ("Lei Complementar") que nunca casaria.
TIPOS_DOCUMENTO = [
    "Lei", "Decreto", "Portaria Interministerial", "Resolução",
    "Nota Técnica", "Nota Técnica Conjunta",
]
ANOS_DISPONIVEIS = [2021, 2023, 2025, 2026]


# ── 1. roteador (código da equipe) ─────────────────────────────────────────
def rotear(pergunta: str, llm: ClienteLLM) -> dict[str, Any]:
    """Classificador binário. Se reprovar, o pipeline para aqui."""
    return roteador_de_intencao(pergunta, llm)


# ── 2. reescrita da consulta ───────────────────────────────────────────────
PROMPT_REESCRITA = """Você reescreve perguntas de leigos na linguagem das normas do Fundeb e do VAAR.

Regras:
- Troque vocabulário coloquial por termos usados em leis, decretos, portarias e notas técnicas do MEC, FNDE e Inep.
- Preserve todo número, sigla, artigo ou ano que a pergunta já traga. Não invente nenhum.
- Devolva UMA frase, sem explicações, sem aspas.

Exemplo:
Pergunta: qual a multa se o município atrasar o repasse
Reescrita: sanções administrativas e penalidades aplicáveis ao ente federado por inadimplemento no repasse dos recursos do Fundeb

Pergunta: {pergunta}
Reescrita:"""


def reescrever(pergunta: str, llm: ClienteLLM) -> str:
    texto = llm.gerar_texto(PROMPT_REESCRITA.format(pergunta=pergunta),
                            max_tokens=256, effort="low")
    texto = texto.strip().strip('"').strip()
    # Se o modelo devolver vazio ou algo absurdo, a pergunta original segue.
    return texto if 10 <= len(texto) <= 600 else pergunta


# ── 3. extração de filtros ─────────────────────────────────────────────────
PROMPT_FILTROS = """Extraia da pergunta os filtros de metadado que ela pede EXPLICITAMENTE.

Campos possíveis:
- "ano": inteiro, apenas se a pergunta citar um ano. Valores existentes: {anos}.
- "tipo_documento": apenas se a pergunta citar o tipo. Valores existentes: {tipos}.

Se a pergunta não citar o campo, use null. Não deduza: "regras de 2026" pede ano 2026; "quais condicionalidades" não pede ano nenhum.

Responda SOMENTE com JSON no formato {{"ano": null, "tipo_documento": null}}.

Pergunta: {pergunta}"""


def extrair_filtros(pergunta: str, llm: ClienteLLM) -> dict[str, Any]:
    bruto = llm.gerar_texto(
        PROMPT_FILTROS.format(pergunta=pergunta, anos=ANOS_DISPONIVEIS,
                              tipos=TIPOS_DOCUMENTO),
        max_tokens=128, effort="low",
    )
    dados = _extrair_json(bruto) or {}
    filtros: dict[str, Any] = {}

    ano = dados.get("ano")
    if isinstance(ano, int) and ano in ANOS_DISPONIVEIS:
        filtros["ano"] = ano
    elif isinstance(ano, str) and ano.isdigit() and int(ano) in ANOS_DISPONIVEIS:
        filtros["ano"] = int(ano)

    tipo = dados.get("tipo_documento")
    if isinstance(tipo, str):
        casado = next((t for t in TIPOS_DOCUMENTO if t.lower() == tipo.strip().lower()), None)
        if casado:
            filtros["tipo_documento"] = casado
    return filtros


# ── 4. HyDE (código da equipe) ─────────────────────────────────────────────
def hyde(pergunta: str, llm: ClienteLLM) -> str:
    return gerar_documento_hyde(pergunta, llm)


# ── 5. geração ancorada (código da equipe) ─────────────────────────────────
def gerar(pergunta: str, documentos: list[dict[str, Any]], llm: ClienteLLM) -> str:
    return gerar_resposta_final(pergunta, documentos, llm)


# ── 6. avaliação de factualidade (código da equipe) ────────────────────────
class _LLMSoNumero:
    """Proxy que entrega a `avaliar_factualidade` só o primeiro número da
    resposta, já limitado a [0, 1]. Assim o `float()` da função original
    nunca quebra, e a chamada ao modelo continua sendo uma só."""

    def __init__(self, llm: ClienteLLM) -> None:
        self._llm = llm
        self.sem_nota = False

    def gerar_texto(self, prompt: str, **kw: Any) -> str:
        bruto = self._llm.gerar_texto(prompt, max_tokens=16, effort="low")
        m = re.search(r"\d+(?:[.,]\d+)?", bruto)
        if not m:
            self.sem_nota = True
            return "0"
        valor = float(m.group(0).replace(",", "."))
        return str(max(0.0, min(1.0, valor)))


def avaliar(resposta: str, documentos: list[dict[str, Any]], llm: ClienteLLM) -> float | None:
    """Chama `scripts/avaliacao_metricas.avaliar_factualidade` como está.
    Devolve None quando o modelo não escreve número nenhum."""
    contexto = [
        {"titulo": d.get("titulo"), "ano": d.get("ano"), "texto": d.get("texto")}
        for d in documentos
    ]
    proxy = _LLMSoNumero(llm)
    nota = avaliar_factualidade(resposta, contexto, proxy)
    return None if proxy.sem_nota else nota


# ── util ───────────────────────────────────────────────────────────────────
def _extrair_json(texto: str) -> dict[str, Any] | None:
    """Acha o primeiro objeto JSON num texto que pode vir com cercas ou prosa."""
    texto = texto.strip()
    texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto, flags=re.S)
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*?\}", texto, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None
