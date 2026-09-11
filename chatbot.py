"""
chatbot.py
Chatbot do RAG VAAR Nacional com rastro do pipeline.

    streamlit run chatbot.py

Três abas:
  Chat        pergunta e resposta, com as fontes citadas
  Pipeline    por onde cada pergunta passou e onde parou
  Avaliação   desfechos agregados e factualidade

Credenciais entram pela barra lateral e ficam só na sessão. Nada é gravado em
disco. Se existirem no .env ou no ambiente, os campos já vêm preenchidos.
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

try:
    from dotenv import load_dotenv
    load_dotenv(RAIZ / ".env")
except ImportError:
    pass

from src.pipeline import (  # noqa: E402
    ANTES_DO_BANCO,
    DESFECHO_DESCARTADA,
    DESFECHO_ERRO,
    DESFECHO_RESPONDIDA,
    DESFECHO_SEM_CONTEXTO,
    ESTAGIOS,
    MODOS_BUSCA,
    NO_BANCO,
    PROVEDORES,
    ArmazemPais,
    ConfigPipeline,
    Pipeline,
    Recuperador,
    Trace,
    carregar_embedder,
    carregar_reranker,
    carregar_vocabulario,
    criar_cliente,
)
from src.embedding.qwen import DEFAULT_TASK  # noqa: E402
from src.interface import (  # noqa: E402
    AVATAR_RESPOSTA,
    ICONE_AVALIACAO,
    ICONE_CHAT,
    ICONE_PIPELINE,
    aplicar_tema,
    cabecalho,
    rodape,
)

st.set_page_config(page_title="RAG VAAR Nacional", page_icon="⚖️", layout="wide")
aplicar_tema()
cabecalho("RAG ", "VAAR Nacional",
          "Assistente sobre a legislação do Fundeb e da complementação VAAR, "
          "com respostas ancoradas em normas oficiais.")

CAMINHO_VOCAB = RAIZ / "data" / "vocabulario_esparso.json"
CAMINHO_CHUNKS = RAIZ / "data" / "chunks" / "chunks_fatec_rag.jsonl"

ICONE_STATUS = {"ok": "✅", "pulado": "⏭️", "parou": "🛑", "erro": "❌", "nao_alcancado": "⬜"}
ROTULO_DESFECHO = {
    DESFECHO_RESPONDIDA: ("Respondida", "green"),
    DESFECHO_DESCARTADA: ("Barrada antes do banco", "orange"),
    DESFECHO_SEM_CONTEXTO: ("Sem contexto, parou antes do LLM", "orange"),
    DESFECHO_ERRO: ("Erro", "red"),
}


# ── recursos pesados, carregados uma vez por processo ───────────────────────
@st.cache_resource(show_spinner="Carregando o Qwen3-Embedding-0.6B (1,2 GB na primeira vez)...")
def _embedder():
    return carregar_embedder()


@st.cache_resource(show_spinner="Carregando o reranker bge-reranker-v2-m3 (2,2 GB na primeira vez)...")
def _reranker():
    return carregar_reranker()


@st.cache_resource
def _vocabulario():
    return carregar_vocabulario(CAMINHO_VOCAB)


@st.cache_resource
def _pais():
    return ArmazemPais(CAMINHO_CHUNKS)


# ── estado da sessão ────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("mensagens", [])      # [{"role", "content", "trace_idx"}]
ss.setdefault("traces", [])         # [Trace]
ss.setdefault("conexao_llm", None)  # (ok, msg)
ss.setdefault("conexao_qdrant", None)


# ── barra lateral ───────────────────────────────────────────────────────────
with st.sidebar:
    # O nome do projeto já está no cabeçalho; aqui só a função da barra.
    st.markdown("### Configuração")

    st.subheader("Banco vetorial (Qdrant)")
    qdrant_url = st.text_input("Endpoint do cluster", value=os.getenv("QDRANT_URL", ""),
                               placeholder="https://....cloud.qdrant.io")
    qdrant_key = st.text_input("API key do Qdrant", type="password",
                               value=os.getenv("QDRANT_API_KEY", ""))
    colecao = st.text_input("Coleção", value="vaar_rag")

    st.subheader("LLM")
    # Padrão: LLM_PROVEDOR do .env; senão Anthropic se houver chave dela;
    # senão Groq, que tem plano gratuito e basta para o projeto.
    _padrao = os.getenv("LLM_PROVEDOR") or ("anthropic" if os.getenv("ANTHROPIC_API_KEY") else "groq")
    _ordem = list(PROVEDORES)
    provedor = st.selectbox("Provedor", _ordem,
                            index=_ordem.index(_padrao) if _padrao in _ordem else 0,
                            format_func=lambda p: PROVEDORES[p]["rotulo"])
    _info = PROVEDORES[provedor]
    _chave_env = os.getenv("ANTHROPIC_API_KEY", "") if provedor == "anthropic" else os.getenv("LLM_API_KEY", "")
    llm_key = st.text_input("API key" + ("" if _info["precisa_chave"] else " (não usada)"),
                            type="password", value=_chave_env, key=f"key_{provedor}",
                            disabled=not _info["precisa_chave"])
    llm_modelo = st.text_input("Modelo", value=os.getenv("LLM_MODELO") or _info["modelo_sugerido"],
                               key=f"modelo_{provedor}")
    llm_base_url = _info["base_url"]
    if provedor == "outro":
        llm_base_url = st.text_input("base_url", value=os.getenv("LLM_BASE_URL", ""),
                                     placeholder="https://.../v1")
    st.caption(_info["dica"])

    if st.button("Testar conexões", use_container_width=True):
        with st.spinner("testando..."):
            try:
                ss.conexao_llm = criar_cliente(provedor, llm_key, llm_modelo, llm_base_url).testar_conexao()
            except Exception as exc:  # noqa: BLE001
                ss.conexao_llm = (False, str(exc))
            try:
                rec = Recuperador(qdrant_url=qdrant_url, qdrant_api_key=qdrant_key,
                                  colecao=colecao, embedder=None, vocabulario={})
                ss.conexao_qdrant = rec.testar_conexao()
            except Exception as exc:  # noqa: BLE001
                ss.conexao_qdrant = (False, str(exc)[:160])

    for rotulo, estado in (("LLM", ss.conexao_llm), ("Qdrant", ss.conexao_qdrant)):
        if estado is not None:
            (st.success if estado[0] else st.error)(f"{rotulo}: {estado[1]}")

    st.subheader("Busca")
    # Medido nesta máquina: o Qdrant devolve a busca híbrida em 17 ms; o Qwen
    # gasta 64 ms por token para vetorizar a consulta. Quem não tem GPU pode
    # trocar a busca semântica por BM25 puro e responder na hora.
    _modos = list(MODOS_BUSCA)
    modo_busca = st.selectbox("Modo", _modos, format_func=lambda m: MODOS_BUSCA[m])
    if modo_busca == "esparsa":
        st.caption("Não carrega o Qwen: responde em milissegundos, mas só acha "
                   "o que casa por palavra.")
    else:
        st.caption("O vetor denso é calculado na CPU e é a etapa mais cara do "
                   "pipeline. Desligue o HyDE para encurtar o texto a vetorizar.")

    st.subheader("Estágios do pipeline")
    usar_reescrita = st.toggle("Reescrita da consulta", value=True)
    usar_filtros = st.toggle("Extração de filtros por metadado", value=True)
    usar_hyde = st.toggle("HyDE (documento hipotético)", value=True,
                          help="Melhora a busca semântica, mas o parágrafo gerado "
                               "é longo e vetorizá-lo na CPU custa dezenas de segundos.")
    usar_rerank = st.toggle("Reranking com cross-encoder", value=False,
                            help="Baixa 2,2 GB na primeira vez e é lento em CPU.")
    usar_pais = st.toggle("Expandir para o chunk pai", value=True)
    usar_avaliacao = st.toggle("Avaliar factualidade", value=True)

    candidatos = st.slider("Candidatos da busca híbrida", 5, 50, 20, 5)
    top_k = st.slider("Trechos enviados ao LLM", 1, 10, 5)
    limite_contexto = st.slider("Tamanho do contexto (caracteres)", 2_000, 40_000, 10_000, 1_000,
                                help="Uma página de tabela da Portaria 14 tem 24 mil caracteres. "
                                     "O plano gratuito da Groq aceita 8.000 tokens por minuto, e o "
                                     "pipeline gasta esse orçamento duas vezes: geração e avaliação.")

    if st.button("Limpar conversa", use_container_width=True):
        ss.mensagens, ss.traces = [], []
        st.rerun()

credenciais_ok = bool(qdrant_url and qdrant_key and llm_modelo
                      and (llm_key or not PROVEDORES[provedor]["precisa_chave"]))


def montar_pipeline() -> Pipeline:
    llm = criar_cliente(provedor, llm_key, llm_modelo, llm_base_url)
    rec = Recuperador(
        qdrant_url=qdrant_url, qdrant_api_key=qdrant_key, colecao=colecao,
        # Em modo esparso o Qwen não é usado, então nem vale baixar 1,2 GB.
        embedder=_embedder() if modo_busca != "esparsa" else None,
        vocabulario=_vocabulario(),
        pais=_pais() if usar_pais else None,
        reranker=_reranker() if usar_rerank else None,
        tarefa_embedding=DEFAULT_TASK,
    )
    cfg = ConfigPipeline(
        usar_reescrita=usar_reescrita, usar_filtros=usar_filtros, usar_hyde=usar_hyde,
        usar_rerank=usar_rerank, usar_contexto_pai=usar_pais, usar_avaliacao=usar_avaliacao,
        candidatos=candidatos, top_k=top_k, modo_busca=modo_busca,
        limite_contexto_chars=limite_contexto,
    )
    return Pipeline(llm, rec, cfg)


# ── abas ────────────────────────────────────────────────────────────────────
aba_chat, aba_pipe, aba_aval = st.tabs([
    f"{ICONE_CHAT} Chat",
    f"{ICONE_PIPELINE} Pipeline",
    f"{ICONE_AVALIACAO} Avaliação",
])


# ── chat ────────────────────────────────────────────────────────────────────
def _pergunta_na_direita(texto: str) -> None:
    """Balão da pergunta, alinhado à direita. O texto é escapado porque vem
    digitado pelo usuário e vai para dentro de HTML."""
    st.markdown(f'<div class="govbr-pergunta">{html.escape(texto)}</div>',
                unsafe_allow_html=True)


def _legenda(tr: Trace) -> str:
    """Linha de status sob a resposta: desfecho, factualidade e tempo."""
    rot, cor = ROTULO_DESFECHO.get(tr.desfecho, ("?", "gray"))
    nota = "" if tr.score_factualidade is None else f" · factualidade {tr.score_factualidade:.1f}"
    return f":{cor}[{rot}]{nota} · {tr.duracao_total_ms()/1000:.1f}s"


def _render_fontes(trace: Trace) -> None:
    if not trace.fontes:
        return
    with st.expander(f"Fontes ({len(trace.fontes)})"):
        for i, f in enumerate(trace.fontes, 1):
            marca = " · página inteira" if f.get("expandido") else ""
            st.markdown(f"**[{i}] {f.get('titulo')}** · {f.get('tipo_documento')} {f.get('ano')} "
                        f"· pág. {f.get('page')}{marca}")
            st.caption((f.get("trecho") or "").replace("\n", " ") + "...")


with aba_chat:
    if not credenciais_ok:
        st.info("Preencha o endpoint e a API key do Qdrant e escolha o provedor de LLM na barra lateral.")

    for m in ss.mensagens:
        if m["role"] == "user":
            _pergunta_na_direita(m["content"])
            continue
        with st.chat_message("assistant", avatar=AVATAR_RESPOSTA):
            st.markdown(m["content"])
            if m.get("trace_idx") is not None:
                tr = ss.traces[m["trace_idx"]]
                st.caption(_legenda(tr))
                _render_fontes(tr)

    # Enquanto não há conversa, este bloco empurra o campo para o meio da tela.
    # Fica num placeholder porque a primeira pergunta só é conhecida depois do
    # st.chat_input, lá embaixo: aí ele é esvaziado na mesma passada.
    abertura = st.empty()
    if not ss.mensagens:
        abertura.markdown(
            '<div class="govbr-abertura">'
            '<span class="govbr-convite">O que você quer saber sobre o VAAR?</span>'
            '<span>Pergunte em linguagem comum. A resposta cita a norma de onde veio.</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    # A troca ao vivo é desenhada AQUI, antes do campo, para que a ordem na
    # tela continue sendo conversa -> campo mesmo antes do próximo rerun.
    ao_vivo = st.container()
    pergunta = st.chat_input("Pergunte sobre o Fundeb ou o VAAR", disabled=not credenciais_ok)
    if pergunta:
        abertura.empty()
        ss.mensagens.append({"role": "user", "content": pergunta})
        with ao_vivo:
            _pergunta_na_direita(pergunta)
        with ao_vivo, st.chat_message("assistant", avatar=AVATAR_RESPOSTA):
            espera = st.empty()
            corpo = st.container()
            status = st.empty()
            espera.markdown("_passando pelo pipeline..._")
            estado = {"mostrada": False}

            def ao_responder(tr: Trace) -> None:
                """Chamado pelo pipeline assim que a resposta existe, ANTES da
                avaliação de factualidade. O juiz relê todo o contexto e leva
                segundos sem mudar uma vírgula do texto, então quem perguntou
                já começa a ler enquanto a nota é calculada."""
                estado["mostrada"] = True
                espera.empty()
                with corpo:
                    st.markdown(tr.resposta or "Sem resposta.")
                    _render_fontes(tr)
                status.caption("avaliando a factualidade...")

            try:
                trace = montar_pipeline().executar(pergunta, ao_responder=ao_responder)
            except Exception as exc:  # noqa: BLE001
                trace = Trace(pergunta=pergunta)
                trace.desfecho = DESFECHO_ERRO
                trace.erro = f"{type(exc).__name__}: {exc}"
                trace.resposta = f"Falha ao montar o pipeline: {trace.erro}"

            espera.empty()
            texto = trace.resposta or "Sem resposta."
            if not estado["mostrada"]:      # parou antes da geração, ou falhou
                with corpo:
                    st.markdown(texto)
                    _render_fontes(trace)
            status.caption(_legenda(trace))
            ss.traces.append(trace)
            ss.mensagens.append({"role": "assistant", "content": texto,
                                 "trace_idx": len(ss.traces) - 1})


# ── pipeline ────────────────────────────────────────────────────────────────
def _render_trace(tr: Trace) -> None:
    rot, cor = ROTULO_DESFECHO.get(tr.desfecho, ("?", "gray"))
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.markdown(f"**Desfecho:** :{cor}[{rot}]")
    c2.markdown(f"**Parou em:** `{tr.onde_parou()}`")
    c3.markdown(f"**Tempo total:** {tr.duracao_total_ms()/1000:.1f}s")

    banco = "chegou" if tr.chegou_ao_banco() else "não chegou"
    llm_final = "chegou" if tr.chegou_ao_llm_final() else "não chegou"
    st.caption(f"Banco vetorial: **{banco}** · LLM de geração: **{llm_final}**")

    alcancados = {e.nome: e for e in tr.etapas}
    linhas = []
    for nome, rotulo in ESTAGIOS:
        e = alcancados.get(nome)
        if e is None:
            status, dur, det = "nao_alcancado", "", "não alcançado"
        else:
            status, dur, det = e.status, f"{e.duracao_ms} ms", e.detalhe
        fase = "antes do banco" if nome in ANTES_DO_BANCO else ("banco" if nome in NO_BANCO else "LLM final")
        linhas.append({"": ICONE_STATUS[status], "Estágio": rotulo, "Fase": fase,
                       "Tempo": dur, "Detalhe": det})
    st.dataframe(linhas, use_container_width=True, hide_index=True,
                 column_config={"Detalhe": st.column_config.TextColumn(width="large")})

    e_busca = alcancados.get("busca")
    if e_busca and e_busca.dados.get("candidatos"):
        with st.expander(f"Candidatos da busca ({e_busca.dados['n_candidatos']})"):
            st.dataframe(
                [{"Título": c["titulo"], "Ano": c["ano"], "Pág.": c["page"],
                  "RRF": round(c["score_rrf"] or 0, 4),
                  "Rerank": round(c["score_rerank"], 3) if c.get("score_rerank") is not None else None,
                  "Trecho": (c["trecho"] or "").replace("\n", " ")}
                 for c in e_busca.dados["candidatos"]],
                use_container_width=True, hide_index=True,
            )
    e_hyde = alcancados.get("hyde")
    if e_hyde and e_hyde.dados.get("documento_hipotetico"):
        with st.expander("Documento hipotético (HyDE) que virou vetor"):
            st.write(e_hyde.dados["documento_hipotetico"])
    if tr.erro:
        st.error(tr.erro)


with aba_pipe:
    st.subheader("Por onde cada pergunta passou")
    st.caption(
        "Fluxo: roteador → reescrita → filtros → HyDE → busca híbrida → rerank → chunk pai → "
        "geração → avaliação. Os quatro primeiros acontecem **antes do banco vetorial**; "
        "se o roteador barrar, o banco nem é consultado."
    )
    if not ss.traces:
        st.info("Nenhuma pergunta ainda.")
    for i, tr in reversed(list(enumerate(ss.traces))):
        rot, _ = ROTULO_DESFECHO.get(tr.desfecho, ("?", "gray"))
        with st.expander(f"#{i+1} · {tr.pergunta[:80]} · {rot}", expanded=(i == len(ss.traces) - 1)):
            _render_trace(tr)


# ── avaliação ───────────────────────────────────────────────────────────────
with aba_aval:
    st.subheader("Desfechos e factualidade")
    if not ss.traces:
        st.info("Nenhuma pergunta ainda.")
    else:
        total = len(ss.traces)
        cont = {k: sum(1 for t in ss.traces if t.desfecho == k) for k in ROTULO_DESFECHO}
        notas = [t.score_factualidade for t in ss.traces if t.score_factualidade is not None]
        c = st.columns(5)
        c[0].metric("Perguntas", total)
        c[1].metric("Respondidas", cont[DESFECHO_RESPONDIDA])
        c[2].metric("Barradas antes do banco", cont[DESFECHO_DESCARTADA])
        c[3].metric("Sem contexto", cont[DESFECHO_SEM_CONTEXTO])
        c[4].metric("Factualidade média", f"{sum(notas)/len(notas):.2f}" if notas else "—")

        st.dataframe(
            [{"#": i + 1, "Pergunta": t.pergunta, "Desfecho": ROTULO_DESFECHO.get(t.desfecho, ("?",))[0],
              "Parou em": t.onde_parou(), "Fontes": len(t.fontes),
              "Factualidade": t.score_factualidade, "Tempo (s)": round(t.duracao_total_ms() / 1000, 1)}
             for i, t in enumerate(ss.traces)],
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "A factualidade é um juiz por LLM: 1.0 = tudo ancorado no contexto, 0.0 = há "
            "informação externa. Ela mede a geração, não a recuperação. Recall@k e MRR "
            "precisam de um conjunto de perguntas anotadas, que ainda não existe."
        )
