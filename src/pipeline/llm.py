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

import re
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
        "modelo_sugerido": "openai/gpt-oss-120b",
        "precisa_chave": True,
        "dica": "console.groq.com; a lista de modelos muda com frequência, "
                "o botão Testar conexões mostra os disponíveis",
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

        extra: dict = {}
        if self._raciocina():
            # Em modelos com raciocínio (gpt-oss, qwen3, o-series) o limite de
            # tokens inclui o raciocínio. Um teto de 16 tokens, suficiente para
            # "1.0" num modelo comum, devolveria content vazio. Por isso o piso,
            # e o esforço baixo para não gastar tokens pensando numa nota.
            max_tokens = max(max_tokens, 1024)
            extra["extra_body"] = {"reasoning_effort": "low"}

        # Modelos novos da OpenAI só aceitam max_completion_tokens; a maioria
        # dos servidores compatíveis só aceita max_tokens. Tenta o mais comum
        # e cai para o outro se a API reclamar do nome do parâmetro.
        try:
            r = self._client.chat.completions.create(
                model=self.modelo, messages=mensagens, max_tokens=max_tokens,
                temperature=0, **extra)
        except self._openai.APIStatusError as e:
            if e.status_code == 429:
                raise RuntimeError(_explicar_429(str(e))) from e
            if e.status_code == 413 or "too large" in str(e).lower():
                # Groq gratuito: 8.000 tokens por minuto. Vale mais avisar o
                # que reduzir do que repassar o JSON cru do provedor.
                raise RuntimeError(
                    "requisição grande demais para o provedor. Reduza o tamanho do "
                    "contexto na barra lateral ou o número de trechos enviados ao LLM."
                ) from e
            if not isinstance(e, self._openai.BadRequestError):
                raise
            msg = str(e)
            if "reasoning_effort" in msg and extra:
                extra = {}
                r = self._client.chat.completions.create(
                    model=self.modelo, messages=mensagens, max_tokens=max_tokens, temperature=0)
            elif "max_completion_tokens" in msg:
                r = self._client.chat.completions.create(
                    model=self.modelo, messages=mensagens, max_completion_tokens=max_tokens, **extra)
            else:
                raise

        if r.usage is not None:
            self.ultimo_uso = {"input_tokens": r.usage.prompt_tokens,
                               "output_tokens": r.usage.completion_tokens}
        escolha = r.choices[0]
        if escolha is None:
            raise RuntimeError("o provedor não devolveu nenhuma escolha")
        if getattr(escolha, "finish_reason", None) == "content_filter":
            raise RespostaRecusada("o provedor bloqueou a resposta por filtro de conteúdo")
        return (escolha.message.content or "").strip()

    _MARCAS_RACIOCINIO = ("gpt-oss", "qwen3", "deepseek-r", "gpt-5", "o1", "o3", "o4")

    def _raciocina(self) -> bool:
        nome = self.modelo.lower().rsplit("/", 1)[-1]        # tira "openai/" etc.
        return any(nome.startswith(p) for p in self._MARCAS_RACIOCINIO)

    def listar_modelos(self) -> list[str]:
        """Modelos de texto que a chave enxerga, sem os de guarda e de áudio.
        Os catálogos mudam (a Groq tirou o llama-3.3 do ar em 2026), então a
        interface mostra isso quando o modelo pedido não existe."""
        ids = sorted(m.id for m in self._client.models.list().data)
        ruins = ("guard", "whisper", "orpheus", "tts", "embed")
        return [i for i in ids if not any(r in i.lower() for r in ruins)]

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
            try:
                disponiveis = ", ".join(self.listar_modelos()[:8])
            except Exception:  # noqa: BLE001
                disponiveis = ""
            return False, (f"modelo '{self.modelo}' não encontrado em {rot}"
                           + (f"; disponíveis: {disponiveis}" if disponiveis else "; confira o nome no painel"))
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


# ── mensagens ──────────────────────────────────────────────────────────────
def _explicar_429(bruto: str) -> str:
    """Traduz o 429 do provedor para algo acionável.

    O plano gratuito da Groq tem dois tetos: por minuto (TPM) e por dia (TPD).
    São coisas muito diferentes: o primeiro passa em segundos, o segundo só
    vira na virada da janela de 24 horas. A mensagem crua não deixa isso claro.
    """
    espera = re.search(r"try again in ([\dhms.]+)", bruto, re.I)
    quando = f" Tente de novo em {espera.group(1).rstrip('.')}." if espera else ""

    if "per day" in bruto.lower() or "(TPD)" in bruto:
        return (
            "limite DIÁRIO de tokens do provedor atingido." + quando +
            " Enquanto isso: troque o modelo na barra lateral, porque a cota é "
            "contada por modelo, ou desligue a avaliação de factualidade, que "
            "sozinha consome quase metade dos tokens de cada pergunta."
        )
    if "per minute" in bruto.lower() or "(TPM)" in bruto:
        return (
            "limite POR MINUTO de tokens do provedor atingido." + quando +
            " Uma pergunta com geração e avaliação gasta cerca de 7.500 tokens, "
            "perto do teto de 8.000 por minuto do plano gratuito: espere um "
            "pouco entre uma pergunta e outra, ou reduza o tamanho do contexto."
        )
    return "limite de requisições do provedor atingido." + quando


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
