"""Cliente de LLM com a interface que os scripts da equipe esperam.

Os scripts em `scripts/` chamam `cliente_llm.gerar_texto(prompt)` e recebem
uma string. Esta é a única superfície que eles conhecem, então qualquer
provedor que implemente `ClienteLLM` encaixa no pipeline sem tocar neles.

A implementação concreta usa a API da Anthropic (Claude) pelo SDK oficial.
"""
from __future__ import annotations

from typing import Protocol

MODELO_PADRAO = "claude-opus-5"

MODELOS = {
    "claude-opus-5": "Claude Opus 5 (padrão)",
    "claude-sonnet-5": "Claude Sonnet 5 (mais barato)",
    "claude-haiku-4-5": "Claude Haiku 4.5 (mais rápido)",
}


class ClienteLLM(Protocol):
    def gerar_texto(self, prompt: str, *, max_tokens: int = 1024,
                    effort: str = "low", system: str | None = None) -> str: ...


class RespostaRecusada(RuntimeError):
    """O modelo recusou a requisição por política (stop_reason == 'refusal')."""


class ClienteAnthropic:
    """Implementa `gerar_texto` sobre `anthropic.Anthropic().messages.create`.

    Em Claude Opus 5 o raciocínio adaptativo já vem ligado por padrão, então o
    parâmetro `thinking` é omitido. A profundidade é controlada por `effort`:
    `low` para classificar, reescrever e extrair; `high` para a resposta final.
    """

    def __init__(self, api_key: str, modelo: str = MODELO_PADRAO,
                 timeout: float = 120.0) -> None:
        import anthropic

        if not api_key or not api_key.strip():
            raise ValueError("API key da Anthropic ausente")
        self.modelo = modelo
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key.strip(), timeout=timeout)
        self.ultimo_uso: dict[str, int] = {}

    def gerar_texto(self, prompt: str, *, max_tokens: int = 1024,
                    effort: str = "low", system: str | None = None) -> str:
        kwargs = dict(
            model=self.modelo,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        # Haiku 4.5 não aceita `effort`; os demais modelos da lista aceitam.
        if not self.modelo.startswith("claude-haiku"):
            kwargs["output_config"] = {"effort": effort}

        resposta = self._client.messages.create(**kwargs)

        self.ultimo_uso = {
            "input_tokens": resposta.usage.input_tokens,
            "output_tokens": resposta.usage.output_tokens,
        }

        if resposta.stop_reason == "refusal":
            detalhe = ""
            if resposta.stop_details is not None:
                detalhe = f" ({resposta.stop_details.category}: {resposta.stop_details.explanation})"
            raise RespostaRecusada(f"o modelo recusou a requisição{detalhe}")

        texto = "".join(b.text for b in resposta.content if b.type == "text")
        return texto.strip()

    def testar_conexao(self) -> tuple[bool, str]:
        """Faz uma chamada mínima. Devolve (ok, mensagem legível)."""
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
