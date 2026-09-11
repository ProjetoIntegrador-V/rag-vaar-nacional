"""Identidade visual do chatbot, inspirada no Design System do gov.br.

O que vem de lá são os tokens públicos do design system: a paleta
(azul #1351B4, azul escuro #071D41, amarelo #FFCD07, verde #168821), a
tipografia Raleway, a faixa superior escura, o filete colorido sob o
cabeçalho e o estilo dos botões e das abas.

O que NÃO vem de lá é a marca. Este é um trabalho acadêmico da FATEC Cotia,
não um serviço do governo federal, então a página usa o nome do projeto e diz
isso na faixa superior. Reproduzir o logotipo gov.br aqui faria a página
passar por oficial, o que ela não é.

Referência dos tokens: https://www.gov.br/ds/
"""
from __future__ import annotations

import streamlit as st

# ── tokens ────────────────────────────────────────────────────────────────
AZUL = "#1351B4"
AZUL_MEDIO = "#0C326F"
AZUL_ESCURO = "#071D41"
AMARELO = "#FFCD07"
VERDE = "#168821"
VERMELHO = "#E52207"
CINZA_FUNDO = "#F8F8F8"
CINZA_BORDA = "#E6E6E6"
CINZA_TEXTO = "#555555"

# Ícones do Material Symbols, que o Streamlit resolve em rótulos de aba.
# Escolhidos pelo que cada aba faz, não pelo que ela mostra:
ICONE_CHAT = ":material/chat:"            # balão de conversa
ICONE_PIPELINE = ":material/psychology:"  # cabeça com engrenagens
ICONE_AVALIACAO = ":material/fact_check:"  # documento conferido

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;500;600;700;800&display=swap');

:root {{
  --govbr-azul: {AZUL};
  --govbr-azul-medio: {AZUL_MEDIO};
  --govbr-azul-escuro: {AZUL_ESCURO};
  --govbr-amarelo: {AMARELO};
  --govbr-verde: {VERDE};
  --govbr-cinza-borda: {CINZA_BORDA};
}}

/* A fonte desce por herança. Um seletor amplo como [class*="st-"] alcancaria
   tambem os <span> dos icones do Streamlit, que dependem da fonte Material
   Symbols para virar desenho: eles apareceriam como o texto cru do nome do
   icone ("face", "smart_toy"). */
html, body, [data-testid="stAppViewContainer"] {{
  font-family: 'Raleway', 'Segoe UI', system-ui, sans-serif;
}}
[data-testid="stIconMaterial"], [data-testid="stTooltipIcon"] svg + span,
.material-symbols-rounded, [class*="material-symbols"] {{
  font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
}}

[data-testid="stAppViewContainer"] {{ background: {CINZA_FUNDO}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMain"] .block-container {{ padding-top: 1.2rem; max-width: 1180px; }}

/* ── cabeçalho ───────────────────────────────────────────────────────── */
.govbr-faixa {{
  background: var(--govbr-azul-escuro);
  color: #FFFFFF;
  font-size: .75rem;
  letter-spacing: .02em;
  padding: .4rem 1.2rem;
  border-radius: 6px 6px 0 0;
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}}
.govbr-faixa .govbr-aviso {{ opacity: .85; }}
.govbr-filete {{ height: 5px; display: flex; }}
.govbr-filete span {{ flex: 1; }}
.govbr-filete .v {{ background: var(--govbr-verde); }}
.govbr-filete .a {{ background: var(--govbr-amarelo); }}
.govbr-filete .z {{ background: var(--govbr-azul); }}
.govbr-marca {{
  background: #FFFFFF;
  border: 1px solid var(--govbr-cinza-borda);
  border-top: 0;
  border-radius: 0 0 6px 6px;
  padding: 1rem 1.2rem 1.1rem;
  margin-bottom: 1.2rem;
}}
.govbr-marca h1 {{
  font-size: 1.7rem; font-weight: 800; margin: 0;
  color: var(--govbr-azul); letter-spacing: -.01em;
}}
.govbr-marca h1 em {{ font-style: normal; font-weight: 400; color: var(--govbr-azul-escuro); }}
.govbr-marca p {{ margin: .25rem 0 0; color: {CINZA_TEXTO}; font-size: .92rem; }}

/* ── abas ────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
  gap: .35rem; border-bottom: 1px solid var(--govbr-cinza-borda);
}}
.stTabs [data-baseweb="tab"] {{
  font-weight: 600; color: {CINZA_TEXTO}; padding: .6rem 1.1rem;
  border-radius: 6px 6px 0 0;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
  color: var(--govbr-azul); background: #FFFFFF;
}}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--govbr-azul); height: 3px; }}

/* ── barra lateral ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid var(--govbr-cinza-borda); }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
  color: var(--govbr-azul-escuro); font-size: 1rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .04em;
  border-left: 4px solid var(--govbr-amarelo); padding-left: .5rem;
}}

/* ── botões e campos ─────────────────────────────────────────────────── */
.stButton > button {{
  background: var(--govbr-azul); color: #FFFFFF; border: 0;
  border-radius: 100px; font-weight: 600; padding: .45rem 1.2rem;
}}
.stButton > button:hover {{ background: var(--govbr-azul-medio); color: #FFFFFF; }}
[data-testid="stChatInput"] {{ border: 1px solid var(--govbr-azul); border-radius: 8px; }}

/* ── blocos de conteúdo ──────────────────────────────────────────────── */
[data-testid="stChatMessage"] {{
  background: #FFFFFF; border: 1px solid var(--govbr-cinza-borda);
  border-radius: 8px; padding: .8rem 1rem;
}}
[data-testid="stMetric"] {{
  background: #FFFFFF; border: 1px solid var(--govbr-cinza-borda);
  border-left: 4px solid var(--govbr-azul); border-radius: 6px; padding: .7rem .9rem;
}}
[data-testid="stExpander"] details {{
  background: #FFFFFF; border: 1px solid var(--govbr-cinza-borda); border-radius: 6px;
}}
.govbr-rodape {{
  margin-top: 2rem; padding: .9rem 1.2rem;
  border-top: 3px solid var(--govbr-amarelo); background: var(--govbr-azul-escuro);
  color: #FFFFFF; border-radius: 0 0 6px 6px; font-size: .8rem;
}}
</style>
"""


def aplicar_tema() -> None:
    """Injeta a folha de estilo. Chame uma vez, logo após set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def cabecalho(titulo_forte: str, titulo_leve: str, descricao: str) -> None:
    """Faixa escura + filete verde/amarelo/azul + marca do projeto."""
    st.markdown(
        f"""
        <div class="govbr-faixa">
          <span>FATEC Cotia &middot; Projeto Integrador V</span>
          <span class="govbr-aviso">Projeto acadêmico. Não é um serviço oficial do governo federal.</span>
        </div>
        <div class="govbr-filete"><span class="v"></span><span class="a"></span><span class="z"></span></div>
        <div class="govbr-marca">
          <h1>{titulo_forte}<em>{titulo_leve}</em></h1>
          <p>{descricao}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def rodape(texto: str) -> None:
    st.markdown(f'<div class="govbr-rodape">{texto}</div>', unsafe_allow_html=True)
