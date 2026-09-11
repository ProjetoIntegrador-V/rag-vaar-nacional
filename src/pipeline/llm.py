"""Clientes de LLM com a interface que os scripts da equipe esperam.

Os scripts em `scripts/` chamam `cliente_llm.gerar_texto(prompt)` e recebem
uma string. Essa é a única superfície que eles conhecem, então qualquer
provedor que a implemente encaixa no pipeline sem tocar neles.

Dois clientes concretos:

- `ClienteAnthropic`         API da Anthropic (Claude), pelo SDK oficial.
- `ClienteOpenAICompativel`  qualquer servidor que fale o protocolo Chat
                             Completions da OpenAI: OpenAI, Groq, Google
                             Gemini (endpoint compatível), Ollama local e
                             outros. Um cliente só, muda a `base_url`.

`criar_cliente()` escolhe pelo nome do provedor. `PROVEDORES` traz a URL e
uma sugestão de modelo para cada um; a sugestão é ponto de partida, não
verdade fixa, porque os catálogos mudam. O campo de modelo na interface é
editável por isso.
"""
from __future__ import annotations

from typing import Protocol

# ── catálogo de provedores ─────────────────────────────────────────────────
PROVEDORES: dict[str, dict] = {
    "anthropic": {
        "rotulo": "Anthropic (Claude)",
        "base_url": None,
        "modelo_sugerido": "claude-opus-5",
        "precisa_chave": True,
        "dica": "console.anthropic.com",
    },
    "openai": {
        "rotulo": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "modelo_sugerido": "gpt-4o-mini",
        "precisa_chave": True,
        "dica": "platform.openai.com",
    },
    "groq": {
        "rotulo": "Groq (tem plano gratuito)",
        "base_url": "https://api.groq.com/openai/v1",
        "modelo_sugerido": "llama-3.3-70b-versatile",
        "precisa_chave": True,
        "dica": "console.groq.com; confira o nome do modelo no painel",
    },
    "gemini": {
        "rotulo": "Google Gemini (endpoint compatível)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "modelo_sugerido": "gemini-2.0-flash",
        "precisa_chave": True,
        "dica": "aistudio.google.com; confira o nome do modelo",
    },
    "ollama": {
        "rotulo": "Ollama (local, sem chave)",
        "base_url": "http://localhost:11434/v1",
        "modelo_sugerido": "llama3.1",
        "precisa_chave": False,
        "dica": "precisa do Ollama rodando e do modelo baixado com `ollama pull`",
    },
    "outro": {
        "rotulo": "Outro compatível com OpenAI",
        "base_url": "",
        "modelo_sugerido": "",
        "precisa_chave": True,
        "dica": "informe a base_url terminada em /v1",
    },
}

MODELO_PADRAO = PROVEDORES["anthropic"]["modelo_sugerido"]


class ClienteLLM(Protocol):
    def gerar_texto(self, prompt: str, *, max_tokens: int = 1024,
                    effort: str = "low", system: str | None = None) -> str: ...

    def testar_conexao(self) -> tuple[bool, str]: ...


class RespostaRecusada(RuntimeError):
    """O modelo recusou a requisição por política."""


# ── Anthropic ──────────────────────────────────────────────────────────────
class ClienteAnthropic:
    """`gerar_texto` sobre `anthropic.Anthropic().messages.create`.

    Em Claude Opus 5 o raciocínio adaptativo já vem ligado; `thinking` é
    omitido. A profundidade é controlada por `effort`: `low` para classificar,
    reescrever e extrair; `high` para a resposta final.
    """

    provedor = "anthropic"

    def __init__(self, api_key: str, modelo: str = MODELO_PADRAO,
                 timeout: float = 120.0) -> None:
        import anthropic

        if not (api_key or "").strip():
            raise ValueError("API key da Anthropic ausente")
        self.modelo = modelo
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key.strip(), timeout=timeout)
        self.ultimo_uso: dict[str, int] = {}

    def gerar_texto(self, prompt: str, *, max_tokens: int = 1024,
                    effort: str = "low", system: str | None = None) -> str:
        kwargs = dict(model=self.modelo, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        if not self.modelo.startswith("claude-haiku"):   # Haiku 4.5 não aceita effort
            kwargs["output_config"] = {"effort": effort}

        r = self._client.messages.create(**kwargs)
        self.ultimo_uso = {"input_tokens": r.usage.input_tokens,
                           "output_tokens": r.usage.output_tokens}
        if r.stop_reason == "refusal":
            det = ""
            if r.stop_details is not None:
                det = f" ({r.stop_details.category}: {r.stop_details.explanation})"
            raise RespostaRecusada(f"o modelo recusou a requisição{det}")
        return "".join(b.text for b in r.content if b.type == "text").strip()

    def testar_conexao(self) -> tuple[bool, str]:
        a = self._anthropic
        try:
            self.gerar_texto("Responda apenas: ok", max_tokens=8, effort="low")
            return True, f"conectado a {self.modelo}"
        except a.AuthenticationError:
            return False, "API key inválida"
        except a.PermissionDeniedError:
            return False, "API key sem permissão para este modelo"
        except a.NotFoundError:
            return False, f"modelo '{self.modelo}' não encontrado"
        except a.RateLimitError:
            return False, "limite de requisições atingido; tente em instantes"
        except a.APIStatusError as e:
            return False, f"erro da API ({e.status_code}): {e.message}"
        except a.APIConnectionError:
            return False, "sem conexão com a API da Anthropic"


# ── OpenAI e compatíveis ───────────────────────────────────────────────────
class ClienteOpenAICompativel:
    """`gerar_texto` sobre `client.chat.completions.create`.

    Chat Completions é o protocolo que Groq, Ollama, Gemini (endpoint
    compatível) e a maioria dos servidores locais implementam. Por isso um
    cliente só serve a todos: muda a `base_url`.

    `effort` é ignorado: não há parâmetro universal para isso neste protocolo.
    """

    def __init__(self, api_key: str, modelo: str, base_url: str,
                 provedor: str = "outro", timeout: float = 120.0) -> None:
        import openai

        if not (base_url or "").strip():
            raise ValueError("base_url ausente para o provedor compatível com OpenAI")
        if not (modelo or "").strip():
            raise ValueError("nome do modelo ausente")
        precisa = PROVEDORES.get(provedor, {}).get("precisa_chave", True)
        if precisa and not (api_key or "").strip():
            raise ValueError(f"API key ausente para {PROVEDORES.get(provedor, {}).get('rotulo', provedor)}")

        self.provedor = provedor
        self.modelo = modelo.strip()
        self.base_url = base_url.strip()
        self._openai = openai
        # Ollama e outros servidores locais ignoram a chave, mas o SDK exige
        # uma string não vazia.
        self._client = openai.OpenAI(api_key=(api_key or "").strip() or "sem-chave",
                                     base_url=self.base_url, timeout=timeout)
        self.ultimo_uso: dict[str, int] = {}

    def gerar_texto(self, prompt: str, *, max_tokens: int = 1024,
                    effort: str = "low", system: str | None = None) -> str:
        mensagens = []
        if system:
            mensagens.append({"role": "system", "content": system})
        mensagens.append({"role": "user", "content": prompt})

        # Modelos novos da OpenAI só aceitam max_completion_tokens; a maioria
        # dos servidores compatíveis só aceita max_tokens. Tenta o mais comum
        # e cai para o outro se a API reclamar do nome do parâmetro.
        try:
            r = self._client.chat.completions.create(
                model=self.modelo, messages=mensagens, max_tokens=max_tokens, temperature=0)
        except self._openai.BadRequestError as e:
            if "max_completion_tokens" not in str(e):
                raise
            r = self._client.chat.completions.create(
                model=self.modelo, messages=mensagens, max_completion_tokens=max_tokens)

        if r.usage is not None:
            self.ultimo_uso = {"input_tokens": r.usage.prompt_tokens,
                               "output_tokens": r.usage.completion_tokens}
        escolha = r.choices[0]
        if getattr(escolha, "finish_reason", None) == "content_filter":
            raise RespostaRecusada("o provedor bloqueou a resposta por filtro de conteúdo")
        return (escolha.message.content or "").strip()

    def testar_conexao(self) -> tuple[bool, str]:
        o = self._openai
        rot = PROVEDORES.get(self.provedor, {}).get("rotulo", self.provedor)
        try:
            self.gerar_texto("Responda apenas: ok", max_tokens=8)
            return True, f"conectado a {self.modelo} em {rot}"
        except o.AuthenticationError:
            return False, f"API key inválida para {rot}"
        except o.PermissionDeniedError:
            return False, f"sem permissão para o modelo '{self.modelo}' em {rot}"
        except o.NotFoundError:
            return False, f"modelo '{self.modelo}' não encontrado em {rot}; confira o nome no painel"
        except o.RateLimitError:
            return False, f"limite de requisições em {rot}; tente em instantes"
        except o.APIStatusError as e:
            # Gemini devolve 400 (nao 401) para chave invalida.
            if "api key" in str(e).lower():
                return False, f"API key inválida para {rot}"
            return False, f"erro da API ({e.status_code}) em {rot}: {str(e)[:120]}"
        except o.APIConnectionError:
            if self.provedor == "ollama":
                return False, "Ollama não respondeu em localhost:11434; ele está rodando?"
            return False, f"sem conexão com {self.base_url}"


# ── fábrica ────────────────────────────────────────────────────────────────
def criar_cliente(provedor: str, api_key: str, modelo: str,
                  base_url: str | None = None) -> ClienteLLM:
    """Constrói o cliente certo para o provedor escolhido na interface."""
    if provedor not in PROVEDORES:
        raise ValueError(f"provedor desconhecido: {provedor}")
    if provedor == "anthropic":
        return ClienteAnthropic(api_key, modelo or MODELO_PADRAO)
    url = (base_url or PROVEDORES[provedor]["base_url"] or "").strip()
    return ClienteOpenAICompativel(api_key, modelo, url, provedor=provedor)
