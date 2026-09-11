# RAG VAAR Nacional

Sistema de perguntas e respostas sobre a legislação do **Fundeb** e da
**Complementação da União por resultados (VAAR)**, com respostas ancoradas em
normas oficiais e citação obrigatória da fonte.

Projeto Integrador V.

---

## Começo rápido

Para quem recebeu o arquivo `.env` pronto e só quer ver o chatbot funcionando.
São três passos.

**1. Baixar o projeto**

```bash
git clone https://github.com/ProjetoIntegrador-V/rag-vaar-nacional.git
```

**2. Colocar o `.env` na pasta do projeto**

Copie o arquivo `.env` que você recebeu para dentro da pasta
`rag-vaar-nacional`, a mesma onde está o `chatbot.py`:

```
rag-vaar-nacional/
├── .env            <- o arquivo vai AQUI
├── chatbot.py
├── executar.bat
├── requirements.txt
└── ...
```

**3. Executar**

| Sistema | O que fazer |
|---|---|
| Windows | clique duas vezes em **`executar.bat`** |
| Linux ou macOS | abra o terminal na pasta e rode `bash executar.sh` |

Na primeira execução o script pergunta como instalar:

| Escolha | Tempo | Espaço | O que você tem |
|---|---|---|---|
| **1. Rápido** (padrão) | ~3 min | 460 MB | busca por palavra (BM25), já com citação da norma |
| **2. Completo** | ~10 min | 2 GB | acrescenta a busca semântica com o modelo Qwen |

A diferença inteira é o PyTorch e o modelo de embedding, que só entram na
busca densa. Se ninguém responder em 20 segundos, ele segue no **Rápido**, que
é o suficiente para ver o sistema inteiro funcionando. Para ligar a busca
semântica depois, rode `pip install -r requirements.txt` dentro da pasta e
troque o **Modo** na barra lateral.

Depois disso o script abre o chatbot em http://localhost:8501. Nas execuções
seguintes ele pula a instalação e abre em segundos.

Pronto. Clique em **Testar conexões** na barra lateral: as duas faixas verdes
confirmam que o banco e o modelo responderam. Depois é só perguntar.

> Não é preciso criar conta em lugar nenhum, nem rodar notebook, nem carregar
> documento: as chaves já estão no `.env` e os documentos já estão no banco.

<details>
<summary>Se preferir fazer à mão, sem o script</summary>

```bash
cd rag-vaar-nacional
python -m venv .venv
.venv\Scripts\activate        # Windows. No Linux e macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run chatbot.py
```

</details>

<details>
<summary>Se der errado</summary>

| O que aparece | O que fazer |
|---|---|
| `[ERRO] Python nao foi encontrado` | instale o Python 3.10 ou mais novo de python.org e marque "Add Python to PATH" |
| `[ERRO] O arquivo .env nao esta nesta pasta` | o `.env` foi para o lugar errado; ele vai na mesma pasta do `chatbot.py` |
| o Windows salvou como `.env.txt` | renomeie para `.env`, sem extensão |
| a resposta demora mais de 30 segundos | na barra lateral, em **Busca**, troque o **Modo** para "Só esparsa" |
| `a busca semântica precisa do PyTorch` | você instalou pelo modo Rápido; rode `pip install -r requirements.txt` ou fique no modo "Só esparsa" |

A seção [11. Como rodar](#11-como-rodar) tem o passo a passo detalhado e uma
tabela de erros mais completa.

</details>

---

## Índice

1. [O que o sistema faz](#1-o-que-o-sistema-faz)
2. [Visão geral do pipeline](#2-visão-geral-do-pipeline)
3. [Etapa 1 e 2: coleta e extração](#3-etapa-1-e-2-coleta-e-extração)
4. [Etapa 3: chunking](#4-etapa-3-chunking)
5. [Etapa 4: embeddings](#5-etapa-4-embeddings)
6. [Etapa 4: modelo esparso](#6-etapa-4-modelo-esparso)
7. [Etapa 4: banco vetorial](#7-etapa-4-banco-vetorial)
8. [Etapa 5: recuperação e fusão](#8-etapa-5-recuperação-e-fusão)
9. [Etapa 6: geração](#9-etapa-6-geração)
10. [Etapa 7: avaliação](#10-etapa-7-avaliação)
    [Chatbot](#chatbot)
11. [Como rodar](#11-como-rodar) **(instalação passo a passo, `.env` e chaves)**
12. [Estrutura do repositório](#12-estrutura-do-repositório)
13. [Pendências conhecidas](#13-pendências-conhecidas)
14. [Publicar o chatbot na internet](#14-publicar-o-chatbot-na-internet) **(limites de memória, Secrets)**

---

## 1. O que o sistema faz

O usuário pergunta em linguagem comum, por exemplo "quais critérios habilitam o
município a receber o VAAR", e o sistema responde citando a norma que sustenta a
resposta. Se a informação não estiver no corpus, ele diz que não está, em vez de
inventar.

O corpus são **22 documentos oficiais**: leis, decretos, resoluções da CIF,
portarias interministeriais e notas técnicas do Inep e da SEB.

## 2. Visão geral do pipeline

```
Pergunta
   |
   v
 1  Filtro de intenção ......... fora de escopo? PARA AQUI, o banco nem é consultado
 2  Reescrita .................. vocabulário leigo -> jargão normativo
 3  Extração de filtros ........ ano e tipo de documento -> filtro do Qdrant
 4  HyDE ....................... LLM escreve um parágrafo hipotético
   |
   v
 5  Busca híbrida no Qdrant .... denso (HyDE, Qwen3) + esparso (pergunta), RRF nativo
                                 nada relevante? PARA AQUI, antes do LLM final
 6  Reranking .................. cross-encoder reordena 20 -> 5 (opcional)
 7  Chunk pai .................. sub-chunk -> página inteira de origem
   |
   v
 8  Geração ancorada ........... resposta com citação da fonte
 9  Avaliação .................. nota de factualidade; não bloqueia a resposta
```

Os estágios 1 a 4 acontecem **antes do banco vetorial**. A aba Pipeline do
chatbot mostra, para cada pergunta, até onde ela chegou.

| Etapa | O que faz | Onde está | Estado |
|---|---|---|---|
| 1. Coleta | Reúne as normas | `data/fonte/` | pronto |
| 2. Extração | PDF e HTML para JSON | `data/fonte/fundeb_vaar_atualizado.json` | pronto, 1 falha |
| 3. Chunking | Segmenta em ~600 tokens | `scripts/chunking.py` | pronto |
| 4. Embedding e indexação | Vetoriza e sobe ao Qdrant | `notebooks/02_embeddings_qdrant.ipynb` | pronto |
| 5. Recuperação | Busca híbrida, reranking, chunk pai | `scripts/motor_recuperacao.py` | pronto |
| 6. Geração | Roteamento, reescrita, HyDE, resposta | `src/pipeline/etapas.py` + `scripts/` | pronto |
| 7. Avaliação | Factualidade por LLM | `src/pipeline/etapas.py` | pronto |
| Interface | Chatbot com rastro do pipeline | `chatbot.py` | pronto |

---

## 3. Etapa 1 e 2: coleta e extração

### O que acontece

Cada arquivo é identificado pelo **conteúdo**, não pelo nome do arquivo, porque
os nomes vinham com abreviações e variações que não correspondiam ao título
oficial. Em seguida são extraídos dois níveis de metadados:

- **Primários**: título, ano, tipo documental, número, data, órgão emissor,
  número do processo administrativo, link para a fonte oficial.
- **Semânticos**: assunto, finalidade, dispositivo legal associado ao Fundeb ou
  ao VAAR, situação da norma e alterações relevantes.

O texto é extraído com **PyMuPDF**, página a página, e depois consolidado em um
campo de texto integral. As duas representações convivem: a contínua serve à
indexação, a segmentada por página serve à rastreabilidade.

### Por que assim

**Sem OCR automático.** Os PDFs já tinham camada textual. Aplicar OCR por cima
introduziria erro de reconhecimento sem ganho nenhum. Quando um PDF não tinha
camada textual, a limitação foi registrada no próprio JSON, em vez de tratar a
ausência de texto como ausência de documento.

**O documento original é a fonte prioritária.** Uma tabela de legislações foi
usada para complementar campos, mas nos casos de divergência entre a tabela e o
arquivo oficial as duas versões foram preservadas e a ocorrência registrada em
observações. Nada foi substituído em silêncio.

**Deduplicação por conteúdo**, combinando tipo documental, número e ano. Várias
normas apareciam duas vezes, como anexo e como link, e viraram um registro só.

**JSON hierárquico** em três níveis: identificação, `informacoes_arquivo` e
`conteudo`. A separação entre metadados e conteúdo é o que permite filtrar antes
de buscar.

Detalhamento em [`docs/metodologia-extracao.md`](docs/metodologia-extracao.md).

---

## 4. Etapa 3: chunking

### O que acontece

`scripts/chunking.py` divide cada documento preferencialmente por parágrafo,
respeitando um teto de **600 tokens** com **overlap de 90 tokens (15%)**, e grava
um JSONL onde cada linha é um chunk com seus metadados.

Resultado: **438 chunks** a partir de 21 documentos.

### Por que assim

**Estrutural antes de fixo.** A divisão tenta respeitar o parágrafo, que é a
unidade natural de sentido em texto normativo. O teto de tokens só entra quando o
parágrafo é grande demais.

**600 tokens** fica no meio da faixa de 500 a 800 recomendada. Chunk pequeno
demais perde o contexto, e o leitor não sabe a que artigo o inciso pertence.
Chunk grande demais dilui o vetor e piora a precisão da recuperação, além de
gastar mais token no prompt final.

**Overlap de 15%** evita que uma explicação que começa no fim de um chunk e
termina no início do outro se perca. Overlap maior inflaria a base sem ganho
proporcional.

**Sem chunking semântico.** Usar um LLM para segmentar produziria chunks mais
coerentes, mas adicionaria custo, dependência de modelo e variabilidade entre
execuções. Como o objetivo é um RAG barato e reproduzível, qualquer integrante
roda o mesmo script com os mesmos parâmetros e obtém o mesmo resultado. O
chunking semântico fica como experimento de comparação, não como método
principal.

**Metadados por chunk**, não só por documento: `document_id`, `title`, `page`,
`chunk_id`, `token_count` e o bloco `metadata` com `ano`, `tipo_documento` e
`orgao`. Sem isso, o embedding seria criado sobre um trecho isolado da origem.

Detalhamento em [`docs/chunking.md`](docs/chunking.md).

---

## 5. Etapa 4: embeddings

### O que acontece

Cada chunk vira um vetor de **1024 dimensões** gerado pelo
**Qwen3-Embedding-0.6B**, normalizado para norma L2 igual a 1.

### Por que Qwen3-Embedding-0.6B, e não o E5

A comparação foi feita com o `intfloat/multilingual-e5-large-instruct`, usando o
**MTEB Leaderboard**, no benchmark de português:

| Característica | Qwen3-Embedding-0.6B | multilingual-e5-large-instruct |
|---|---|---|
| MTEB Português | #2 de 240, nota **71,63** | #1 de 240, nota **72,09** |
| Dimensão | 1.024 | 1.024 |
| Máximo de tokens | **32.768** | 514 |
| Parâmetros | 596M | 560M |
| Licença | Apache-2.0 | MIT |

**O Qwen não é superior ao E5 em português.** Ele perde por 0,46 ponto. A escolha
não se apoia no ranking, e sim na **capacidade de entrada**: 32.768 tokens contra
514.

Isso não significa jogar o PDF inteiro no modelo. O chunking continua
necessário. A vantagem é ter margem para **testar diferentes tamanhos de chunk**
sem esbarrar no limite do modelo. Com o E5, um chunk de 800 tokens já não caberia,
o que travaria a comparação experimental prevista na etapa 3.

**Ressalva metodológica:** ranking externo não decide qual modelo é melhor para
este corpus. A validação final exige um benchmark próprio, com Recall@k, MRR ou
nDCG sobre as normas do projeto. Esse benchmark ainda não existe (ver
[pendências](#13-pendências-conhecidas)).

### Três detalhes do modelo que quebram a busca em silêncio

1. **Query e documento são assimétricos.** A consulta leva um prefixo de
   instrução, o documento não. Codificar documento com prefixo de query não gera
   erro, só piora a recuperação. Por isso existem `encode_queries()` e
   `encode_documents()` separados em `src/embedding/qwen.py`.
2. **Pooling é de último token** (EOS), não média. Exige padding à esquerda,
   senão o vetor sai do token de padding.
3. **Vetores normalizados**, então produto interno é igual a cosseno. No Qdrant a
   métrica é `COSINE`, nunca `EUCLID`.

---

## 6. Etapa 4: modelo esparso

### O que acontece

O mesmo texto também vira um **vetor esparso**: o texto é tokenizado em português
e cada termo vira um par índice, frequência. O **IDF é calculado pelo Qdrant**,
por meio do `Modifier.IDF` na configuração da coleção.

### Por que BM25, e não SPLADE ou BGE-M3

| Característica | BM25 | SPLADE | BGE-M3 |
|---|---|---|---|
| Tipo | Algoritmo estatístico | Rede neural esparsa | Híbrido nativo |
| Custo computacional | **zero**, roda no banco | alto, exige GPU/CPU | muito alto |
| Redundância com o Qwen | nenhuma | parcial | total |
| Espaço no banco | mínimo | médio | alto |

**O critério é complementaridade, não sofisticação.** O Qwen já entrega a
compreensão semântica. O modelo esparso precisa preencher apenas a lacuna da
busca exata, e para isso não precisa "pensar".

**SPLADE seria redundante.** Ele carrega uma segunda rede neural na memória para
fazer expansão de vocabulário, que é justamente o que o Qwen já resolve na camada
semântica.

**BGE-M3 seria desperdício.** É um modelo "all-in-one"; usá-lo só para extrair a
camada esparsa obrigaria a rodar dois modelos grandes sobre o mesmo texto, e
tornaria o Qwen redundante no pipeline.

**BM25 é uma fórmula, não um modelo.** Para um corpus onde siglas como VAAT-MIN,
Simec e VAAF e quantias exatas não têm sinônimo, ele age como lupa de precisão
enquanto o Qwen trabalha o contexto.

### Precisão sobre "BM25 nativo do Qdrant"

O Qdrant fornece o **IDF**, a metade estatística do BM25, via `Modifier.IDF`. A
**tokenização é nossa**, em `src/esparso/bm25.py`, e as regras são específicas
deste domínio:

- **Acentos removidos** dos dois lados. Usuário de portal público digita
  "orcamento" com frequência.
- **Ponto, hífen e barra preservados dentro do token**, para não picar `art.14`,
  `3106200-1` e `2025/2026`, que são exatamente as âncoras que importam.
- **Sem stemmer.** Ele aumentaria recall em texto corrido, mas achataria as
  âncoras exatas. Generalização morfológica já é trabalho do lado denso.

### Por que deixar o IDF no banco

A biblioteca `rank_bm25`, usada no protótipo, aplica a fórmula Okapi, cujo IDF
**vai a zero quando o termo aparece em metade dos documentos e fica negativo
acima disso**. Em corpus especializado, palavras legítimas como "fundeb" e
"municipio" caem nessa faixa e desaparecem do resultado lexical.

Teste com três documentos, todos contendo "fundeb", e só um contendo o código
3106200:

```
consulta 'fundeb'           0.1335  doc a, doc b, doc c
consulta '3106200'          0.9808  doc c
consulta 'fundeb 3106200'   1.1144  doc c
```

O termo comum recebe peso baixo, porém **positivo**. Com o Okapi ele receberia
zero e o resultado seria descartado. Mover o IDF para o banco elimina a armadilha
sem esforço.

---

## 7. Etapa 4: banco vetorial

### O que acontece

Uma coleção `vaar_rag` no **Qdrant**, com dois vetores nomeados por ponto:
`denso` (1024, cosseno) e `esparso` (com `Modifier.IDF`), mais o payload com
todos os metadados.

### Por que Qdrant

Seis candidatos avaliados contra os requisitos reais do projeto:

| Critério | Qdrant | Chroma | Pinecone | Weaviate | Milvus | pgvector |
|---|---|---|---|---|---|---|
| Licença | Apache 2.0 | Apache 2.0 | Proprietária | BSD-3 | Apache 2.0 | PostgreSQL |
| Self-host gratuito | sim | sim | **não** | sim | sim | sim |
| Modo embutido | sim | sim | não | não | parcial | não |
| Vetor esparso nativo | sim | **não** | sim | sim | sim | sem rank |
| Fusão nativa | **RRF e DBSF** | **não** | não aval. | RSF e Ranked | RRF e Weighted | **não** |
| IDF no banco | sim | não | não aval. | sim | sim | não |
| Infra para dev | **nenhuma** | nenhuma | conta + rede | Docker | Docker | Postgres |

**Chroma** é o mais fácil de começar e é aí que engana: a busca textual dele são
operadores de filtro, sensíveis a maiúsculas. Filtro restringe o conjunto, não
ordena por relevância. Sem BM25 e sem fusão, ele resolve só metade do problema.

**Pinecone** foi eliminado por reprodutibilidade: não há execução local no plano
gratuito, e a alternativa auto-hospedada é corporativa, com mínimo de 500 dólares
mensais.

**Weaviate e Milvus** são tecnicamente adequados e fazem fusão nativa. Foram
preteridos por custo operacional, não por capacidade: ambos pressupõem servidor
em contêiner. O volume deste corpus está muito abaixo do ponto em que a escala do
Milvus compensaria sua complexidade.

**pgvector** faria sentido se o projeto já rodasse sobre PostgreSQL. Não é o
caso, e ele não traz BM25 nem fusão.

**Ressalva honesta:** a vantagem do Qdrant não é ser tecnicamente superior a
Weaviate ou Milvus. É entregar a mesma fusão nativa sem exigir infraestrutura.

### Nuvem ou local

O comparativo recomendou o **modo local embutido**, com peso grande para
"funcionar offline" e "não depender de conta de terceiro".

O projeto hoje usa o **Qdrant Cloud**, e isso é uma mudança consciente em relação
àquela recomendação. O motivo é prático: a equipe inteira consulta a mesma
coleção sem cada integrante baixar 1,2 GB de modelo e reprocessar os 438 chunks.

O notebook suporta os dois modos com uma variável só, `MODO_QDRANT`. O argumento
de reprodutibilidade continua válido: o professor consegue rodar tudo em modo
local, sem conta nenhuma.

Comparativo completo em
[`docs/escolha-banco-vetorial.docx`](docs/escolha-banco-vetorial.docx).

---

## 8. Etapa 5: recuperação e fusão

### O que acontece

A consulta é vetorizada dos dois jeitos e o Qdrant busca em paralelo nos dois
índices, trazendo 50 candidatos de cada lado, e funde por **RRF**.

### Por que híbrida, e não só densa

Os dois retrievers erram em lugares diferentes:

| Consulta | Denso | Lexical |
|---|---|---|
| "quais critérios habilitam o município" | acerta, entende a paráfrase | erra se a norma escreve "requisitos de elegibilidade" |
| "art. 14 da Lei 14.113" | fraco, o vetor borra o número | **acerta**, casamento exato |
| "Portaria 14/2025" | fraco | **acerta** |

No domínio jurídico a precisão exata é inegociável. Modelos puramente semânticos
lidam mal com número de lei, sigla e numeração de artigo. E o pior é que erram
**em silêncio**: devolvem o trecho errado sem sinalizar nada.

### Por que RRF, e não soma ponderada

O cosseno do Qwen vive num intervalo estreito e alto, tipicamente 0,3 a 0,9. O
score lexical é ilimitado para cima e varia com o tamanho do corpus, a raridade
dos termos e o tamanho da consulta.

Somar os dois exigiria normalizar, e toda normalização (min-max, z-score) depende
do lote de resultados daquela consulta. O peso calibrado numa pergunta não valeria
na seguinte.

RRF descarta o score e usa só a posição:

```
RRF(d) = soma_i  peso_i / (k + rank_i(d))          k = 60
```

Livre de escala, sem nada para calibrar, e não quebra quando um dos lados devolve
poucos resultados.

### Filtro por metadado

`buscar()` aceita filtro por `ano` e por `tipo_documento`. Serve para perguntas
como "o que mudou nas regras de 2026" e evita que o LLM responda com uma norma
superada.

**Armadilha documentada:** numa consulta com fusão, o filtro precisa ir **dentro
de cada `prefetch`**. Passado só no topo, ele não propaga: cada lado traz
candidatos sem filtro e o RRF apenas reordena, então vazam documentos de outros
anos. Isso foi verificado na prática e está comentado no notebook.

Detalhamento em [`docs/retrieval-hibrida.md`](docs/retrieval-hibrida.md).

---

## 9. Etapa 6: geração

Três passos, todos em `scripts/`:

**Filtro de intenção** (`filtro_intencao.py`). Classificador binário: a pergunta
é sobre Fundeb, VAAR, MEC ou Inep? Se não for, o pipeline para antes de consultar
o banco, poupando processamento.

**HyDE** (`geracao_hyde.py`). O LLM escreve um parágrafo hipotético imitando a
linguagem de uma nota técnica, e é **esse texto** que vira vetor, não a pergunta
crua. O motivo é o descompasso de vocabulário: o usuário pergunta "qual a multa
se atrasar o repasse", a norma diz "sanções administrativas por inadimplemento".
O documento hipotético aproxima a consulta do jargão do corpus.

**Geração ancorada** (`geracao_rag_final.py`). Os trechos recuperados são
formatados com título e ano, e o prompt obriga o modelo a responder
**exclusivamente** com base neles, a dizer "a legislação recuperada não contém
essa informação" quando faltar, e a citar o documento no fim.

### Contrato entre recuperação e geração

A função `buscar()` devolve exatamente o formato que o `geracao_rag_final.py`
consome:

```python
[{"titulo": ..., "ano": ..., "texto": ..., "page": ..., "tipo_documento": ...}, ...]
```

Encaixe direto, sem ninguém alterar código:

```python
rota = roteador_de_intencao(pergunta, cliente_llm)
if rota["status"] == "aprovado":
    hipotetico = gerar_documento_hyde(rota["pergunta"], cliente_llm)
    docs = buscar(hipotetico, top_k=5)
    resposta = gerar_resposta_final(rota["pergunta"], docs, cliente_llm)
```

---

## 10. Etapa 7: avaliação

`avaliacao_metricas.py` implementa um **score de factualidade**: um LLM auditor
compara a resposta com o contexto recuperado e devolve 1.0 quando tudo está
ancorado ou 0.0 quando há informação externa.

A avaliação **não bloqueia a resposta ao usuário**. A ideia é rodar em paralelo e
alimentar um painel de qualidade.

Como alternativa às chamadas de API, modelos locais em português, como o
BERTimbau, podem calcular similaridade de cosseno entre a resposta e os
documentos de origem.

**O que falta:** isso mede a geração, não a recuperação. As métricas de
recuperação, Recall@k e MRR, ainda precisam ser implementadas, e dependem de um
conjunto de perguntas anotadas que ainda não existe.

---

## Chatbot

`chatbot.py` é a interface do sistema. Três abas:

| Aba | O que mostra |
|---|---|
| **Chat** | pergunta, resposta e as fontes citadas, com página e tipo de norma |
| **Pipeline** | por onde **cada pergunta** passou: os nove estágios, quanto tempo levou em cada um e, se parou, em qual e por quê |
| **Avaliação** | desfechos agregados e a média de factualidade |

A aba Pipeline responde a pergunta que importa para depuração: **a pergunta
chegou ao banco vetorial? Chegou ao LLM de geração?** Cada trace tem um de
quatro desfechos:

| Desfecho | Onde parou |
|---|---|
| Respondida | passou por tudo |
| Barrada antes do banco | o roteador de intenção reprovou; o Qdrant nem foi consultado |
| Sem contexto | chegou ao banco, nada relevante voltou; parou antes do LLM de geração |
| Erro | exceção em algum estágio, com a mensagem |

### Os nove estágios

```
pergunta
  1 roteador      classificador binário: é Fundeb/VAAR? senão, PARA AQUI
  2 reescrita     vocabulário leigo -> jargão normativo
  3 metadados     extrai ano e tipo de documento -> filtro do Qdrant
  4 hyde          LLM escreve um parágrafo hipotético; é ELE que vira vetor denso
  5 busca         híbrida no Qdrant: denso (HyDE) + esparso (pergunta), RRF nativo
                  se vier vazio, PARA AQUI
  6 rerank        cross-encoder reordena os 20 candidatos, ficam 5
  7 contexto_pai  sub-chunk -> página inteira de origem (small-to-big)
  8 geracao       resposta ancorada, com citação obrigatória
  9 avaliacao     juiz por LLM dá nota de factualidade; não bloqueia a resposta
```

Os estágios reutilizam os scripts da equipe em `scripts/`:

| Estágio | Código |
|---|---|
| 1 roteador | `filtro_intencao.roteador_de_intencao` |
| 4 hyde | `geracao_hyde.gerar_documento_hyde` |
| 5 busca e 6 rerank | `motor_recuperacao.MotorRecuperacaoVAAR` (etapas A, B e C) |
| 8 geracao | `geracao_rag_final.gerar_resposta_final` |
| 9 avaliacao | `avaliacao_metricas.avaliar_factualidade` |

Os estágios 2 (reescrita), 3 (metadados) e 7 (chunk pai) vinham da arquitetura
alvo e foram implementados em `src/pipeline/`. O `motor_recuperacao.py`
recebeu o mínimo para rodar contra a coleção real: imports pelos pacotes do
repo, conexão em nuvem, vocabulário termo -> índice no vetor esparso e filtro
dentro de cada prefetch; o cabeçalho do arquivo lista cada mudança. Cada
estágio pode ser ligado ou desligado na barra lateral, e um estágio desligado
aparece na aba Pipeline como "pulado", não some.

### A resposta sai antes da avaliação

A avaliação de factualidade é a etapa mais lenta depois da busca: o juiz relê
todo o contexto e leva de 5 a 30 segundos. E ela não altera uma vírgula do
texto gerado, porque só atribui uma nota. Fazer o usuário esperar por ela
seria cobrar por um trabalho que não muda a resposta dele.

Por isso `Pipeline.executar` aceita `ao_responder`, um callback chamado assim
que a geração termina e antes de o juiz rodar:

```python
def ao_responder(trace):
    st.markdown(trace.resposta)      # já aparece na tela
    status.caption("avaliando a factualidade...")

trace = pipeline.executar(pergunta, ao_responder=ao_responder)
st.caption(f"factualidade {trace.score_factualidade}")   # chega depois
```

Medido na interface: a resposta aparece aos 18,5 s e a nota chega depois, sem
que o texto mude. Quando a pergunta para antes da geração (barrada no roteador
ou sem contexto) o callback não é chamado, e a interface desenha o desfecho
normalmente.

### Identidade visual

A interface segue o Design System do gov.br (https://www.gov.br/ds/): paleta
azul #1351B4 / #071D41 com amarelo #FFCD07 e verde #168821, tipografia
Raleway, faixa superior escura, filete verde-amarelo-azul e botões em pílula.
Os tokens ficam em `src/interface/govbr.py` e o tema base em
`.streamlit/config.toml`.

O que o projeto **não** usa é a marca gov.br. Este é um trabalho acadêmico da
FATEC Cotia, não um serviço do governo federal, e a faixa superior diz isso.
Reproduzir o logotipo faria a página passar por oficial.

As abas usam Material Symbols, que o Streamlit resolve no rótulo:
`:material/chat:` para o Chat, `:material/psychology:` para o Pipeline e
`:material/fact_check:` para a Avaliação.

### A conversa

A pergunta fica à direita, num balão azul claro; a resposta fica à esquerda,
com a bandeira do Brasil no avatar. O balão da pergunta é HTML próprio, com o
texto escapado, e não `st.chat_message`: alinhar o componente do Streamlit
dependeria da estrutura interna dele, que muda de versão para versão.

O avatar é o arquivo `src/interface/bandeira.svg`, não o emoji 🇧🇷. No Windows
os indicadores regionais não têm desenho na fonte do sistema e o emoji aparece
como as letras "BR".

O campo de pergunta segue o padrão do ChatGPT: enquanto não há conversa ele
fica no meio da tela, sob um convite; depois da primeira pergunta ele desce e
gruda no rodapé enquanto a conversa rola por cima. São duas peças:

- um bloco de abertura de `30vh` que empurra o campo para o meio. Ele vive num
  `st.empty()` porque a primeira pergunta só é conhecida depois do
  `st.chat_input`, lá embaixo: aí o placeholder é esvaziado na mesma passada;
- `position: sticky` no contêiner do campo, em vez de `fixed`. Sticky
  acompanha a largura da coluna sozinho, sem descontar a barra lateral na
  mão, e some junto com a aba quando o usuário troca de aba.

A troca ao vivo é desenhada num `st.container()` criado **antes** do campo,
senão a resposta nova apareceria abaixo dele até o próximo rerun.

### Duas decisões que não são óbvias

**O lado esparso não recebe o HyDE.** O documento hipotético vai só para o
vetor denso. O esparso recebe a pergunta original mais a reescrita, porque as
âncoras exatas ("art. 14", "Portaria 14/2025") estão na pergunta do usuário,
não num parágrafo inventado pelo modelo.

**O reranker vem desligado por padrão.** O `bge-reranker-v2-m3` pesa 2,2 GB e
é lento em CPU. Ligue na barra lateral quando houver GPU ou quando a precisão
da ordenação importar mais que a latência.

### Onde vai o tempo

Medido na máquina de desenvolvimento (4 núcleos, sem GPU), contra o cluster
real com 2.539 pontos:

| Etapa | Tempo |
|---|---|
| Qdrant, busca híbrida com fusão RRF | **17 ms** |
| Qwen, vetorizar a consulta | **64 ms por token** |

O banco não é o gargalo: ele responde em milissegundos. O custo é vetorizar a
consulta localmente, e ele cresce com o tamanho do texto. Como o HyDE escreve
um parágrafo inteiro (cerca de 470 tokens), era esse parágrafo que levava a
busca a mais de 50 segundos. Três medidas cortaram isso:

1. **Usar todos os núcleos.** O torch usava 2 de 4 por padrão; 458 tokens
   caíram de 53 s para 31 s só com isso.
2. **Truncar a consulta em 256 tokens.** Os chunks foram indexados com até
   1.024 tokens, mas a consulta não precisa do mesmo teto: o vetor truncado em
   256 tokens tem cosseno 0,987 contra o vetor do texto inteiro. Mesma direção,
   um terço do tempo.
3. **Modo de busca selecionável**, na barra lateral:

| Modo | O que faz | Latência |
|---|---|---|
| Híbrida | denso + esparso, fundidos por RRF | dominada pelo Qwen |
| Só esparsa | BM25 puro; **não carrega o Qwen** | ~50 ms |
| Só densa | apenas semântica | dominada pelo Qwen |

Com HyDE desligado e busca híbrida, a etapa cai para cerca de 4 s. Em modo
esparso, para 53 ms. Quem tiver GPU não precisa de nada disso.

### Quanto custa uma pergunta

Medido numa pergunta real, com `openai/gpt-oss-120b` na Groq, contando os
tokens que o próprio provedor reporta em cada chamada:

| Etapa | Entrada | Saída | Total |
|---|---:|---:|---:|
| Roteador | 157 | 23 | 180 |
| Reescrita | 230 | 81 | 311 |
| Metadados | 246 | 53 | 299 |
| **Geração** | 2.839 | 438 | **3.277** |
| **Avaliação** | 3.297 | 107 | **3.404** |
| **Total** | 6.769 | 702 | **7.471** |

Dois números saltam:

- **geração e avaliação somam 89% do gasto.** As três primeiras etapas, que
  parecem muitas chamadas, custam 790 tokens juntas. O peso está em quem
  carrega o contexto recuperado;
- **a avaliação sozinha é 46%.** Ela relê todo o contexto para dar uma nota e
  não muda uma vírgula da resposta.

No plano gratuito da Groq isso dá **8.000 tokens por minuto e 200.000 por
dia, contados por modelo**. Ou seja, cerca de **26 perguntas por dia**, e uma
pergunta sozinha quase estoura o teto do minuto.

O que fazer quando o limite chegar:

| Medida | Efeito |
|---|---|
| Desligar **Avaliar factualidade** | quase dobra as perguntas por dia |
| Baixar **Tamanho do contexto** de 10.000 para 6.000 | corta cerca de 30% |
| Trocar o modelo na barra lateral | a cota é por modelo, então o outro está zerado |
| Esperar | o teto por minuto passa em segundos; o do dia, na virada da janela |

A aba **Pipeline** mostra os tokens de cada etapa e a aba **Avaliação** traz o
total da sessão, a média por pergunta e a estimativa de quantas cabem no dia.
Quando o provedor recusa por limite, a mensagem diz se o teto foi o do minuto
ou o do dia, que são coisas bem diferentes.

### Teto de contexto

A expansão para o chunk pai devolve a página inteira, e uma página de tabela
da Portaria 14 tem 24 mil caracteres. Cinco delas somam 120 mil, o que estoura
o limite de qualquer provedor: no plano gratuito da Groq são 8.000 tokens por
minuto, e o pipeline gasta esse orçamento duas vezes, na geração e na
avaliação. O contexto é cortado para caber no teto da barra lateral (10 mil
caracteres por padrão), repartindo o orçamento como uma torneira: quem já cabe
na fatia leva o texto inteiro e devolve a sobra, então um trecho curto nunca é
cortado por causa de um vizinho gigante. Os trechos cortados aparecem
marcados na aba Pipeline.

### Filtro de metadado que não casa com nada

O filtro é um palpite do LLM, e às vezes ele pede uma combinação que não
existe. "Portaria 14 de 2026" vira `ano=2026 + Portaria Interministerial`, mas
a Portaria 14 está indexada como **2025**: foi publicada em dezembro de 2025 e
rege o exercício de 2026. Antes, a busca voltava vazia e a resposta era "a
legislação recuperada não contém essa informação", o que era falso. Agora,
quando o filtro não casa com nada, a busca é repetida sem ele e a aba Pipeline
registra qual filtro foi ignorado. O vetor denso fica em cache, então a
repetição não custa nada.

### Provedor de LLM

O pipeline não depende de um provedor específico. Os scripts da equipe só
conhecem `cliente_llm.gerar_texto(prompt)`, e dois clientes implementam isso
(`src/pipeline/llm.py`):

| Provedor | Como | Chave |
|---|---|---|
| Anthropic (Claude) | SDK oficial `anthropic` | sim |
| OpenAI | protocolo Chat Completions | sim |
| Groq | mesmo protocolo, `base_url` da Groq; tem plano gratuito | sim |
| Google Gemini | endpoint compatível com OpenAI | sim |
| Ollama | mesmo protocolo em `localhost:11434`; roda offline | não |
| Outro | qualquer servidor compatível, informe a `base_url` | depende |

Escolha na barra lateral. O nome do modelo é um campo editável com uma
sugestão, porque os catálogos mudam; confira no painel do provedor.

### Credenciais

Os campos da barra lateral ficam só na sessão. Se `QDRANT_URL`,
`QDRANT_API_KEY`, `LLM_PROVEDOR`, `LLM_API_KEY` e `LLM_MODELO` existirem no
`.env` ou no ambiente, os campos já vêm preenchidos (só `ANTHROPIC_API_KEY`
também funciona). O botão **Testar conexões** valida o banco e o LLM antes da
primeira pergunta.

## 11. Como rodar

Passo a passo do zero até o chatbot respondendo. Tudo acontece dentro da pasta
do repositório, chamada aqui de **RAIZ**: é a pasta que contém o `chatbot.py`.

```
rag-vaar-nacional/        <- RAIZ
├── chatbot.py
├── requirements.txt
├── .env                  <- você vai criar este arquivo (passo 11.4)
├── .env.example
├── data/
├── notebooks/
├── scripts/
└── src/
```

### 11.1 O que você precisa antes

| O quê | Onde consegue | Custa |
|---|---|---|
| Python 3.10 ou mais novo | python.org | grátis |
| Git | git-scm.com | grátis |
| Conta no Qdrant Cloud | cloud.qdrant.io | grátis, 1 GB basta |
| Chave de um provedor de LLM | console.groq.com | grátis |
| Cerca de 3 GB livres em disco | modelos baixados do HuggingFace | |

Confira o Python antes de começar:

```bash
python --version
```

Se aparecer 3.9 ou menor, instale uma versão mais nova. O código usa sintaxe de
tipos que só existe a partir do 3.10.

### 11.2 Baixar o projeto

```bash
git clone https://github.com/ProjetoIntegrador-V/rag-vaar-nacional.git
cd rag-vaar-nacional
```

A partir daqui, **todos os comandos deste guia são executados de dentro desta
pasta**, a RAIZ.

### 11.3 Ambiente virtual e dependências

O ambiente virtual isola as bibliotecas deste projeto das do resto da máquina.

Há dois arquivos de dependências:

| Arquivo | Tamanho | Para quê |
|---|---|---|
| `requirements-minimo.txt` | 460 MB | roda o chatbot em busca esparsa (BM25) |
| `requirements.txt` | 2 GB | tudo, inclusive PyTorch e o modelo de embedding |

A diferença é só o PyTorch e o `sentence-transformers`. Comece pelo mínimo se
quiser ver o sistema rodando logo; o completo pode ser instalado por cima
depois, sem refazer nada.

**Windows, no PowerShell:**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-minimo.txt
```

Se o PowerShell recusar o script com "execução de scripts foi desabilitada",
rode uma vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e tente de
novo. No Prompt de Comando antigo o ativador é `.venv\Scripts\activate.bat`.

**Linux ou macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Deu certo quando o nome `(.venv)` aparece no começo da linha do terminal. Esse
passo baixa cerca de 2 GB (o PyTorch é o maior deles) e leva alguns minutos.

**Toda vez que abrir um terminal novo, ative o ambiente de novo** antes de
rodar qualquer comando do projeto.

### 11.4 O arquivo .env: onde fica e o que escrever

O `.env` guarda as chaves. Ele fica **na RAIZ, ao lado do `chatbot.py`**, no
caminho `rag-vaar-nacional/.env`. Não é dentro de `src/`, nem de `data/`, nem
de `notebooks/`.

Comece copiando o modelo que já vem no repositório:

```bash
cp .env.example .env
```

No Windows, no PowerShell, o comando é `Copy-Item .env.example .env`.

Agora abra `rag-vaar-nacional/.env` em qualquer editor de texto e preencha. O
arquivo inteiro fica assim:

```ini
# ── Qdrant Cloud ────────────────────────────────────────────────────────
QDRANT_URL=https://ab12cd34-5678-90ef-ghij-klmnopqrstuv.sa-east-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=COLE-AQUI-A-CHAVE-DO-QDRANT

# ── Modelo de embedding ─────────────────────────────────────────────────
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B

# ── LLM da etapa de geração ─────────────────────────────────────────────
LLM_PROVEDOR=groq
LLM_API_KEY=COLE-AQUI-A-CHAVE-DO-PROVEDOR
LLM_MODELO=openai/gpt-oss-120b
LLM_BASE_URL=
```

Regras que evitam dor de cabeça:

- **sem aspas** em volta dos valores: `QDRANT_API_KEY=eyJhbGci...`, não
  `QDRANT_API_KEY="eyJhbGci..."`;
- **sem espaço** antes ou depois do `=`;
- **a chave inteira em uma linha só**. A chave do Qdrant tem cerca de 180
  caracteres e o editor pode quebrá-la visualmente; o que não pode é ter uma
  quebra de linha de verdade no meio;
- **o arquivo se chama `.env`**, com o ponto na frente e sem extensão. O Bloco
  de Notas do Windows costuma salvar como `.env.txt`: no diálogo de salvar,
  escolha "Todos os arquivos" em Tipo.

O `.env` está no `.gitignore`, então o Git o ignora. **Nunca comite a chave**:
uma vez no histórico do Git ela vaza, mesmo que o arquivo seja apagado depois.
Se preferir não criar arquivo nenhum, dá para digitar as chaves direto na barra
lateral do chatbot: elas ficam só na sessão e não são gravadas em disco.

### 11.5 De onde vem cada chave

**Qdrant (`QDRANT_URL` e `QDRANT_API_KEY`)**

1. Crie uma conta em https://cloud.qdrant.io.
2. Em **Clusters**, clique em **Create** e escolha o plano gratuito. A região
   `sa-east-1` (São Paulo) é a mais próxima.
3. Quando o cluster ficar verde, copie o **Cluster endpoint**. É a URL que
   termina em `.cloud.qdrant.io` e vai em `QDRANT_URL`. Copie sem barra no
   final e sem a porta.
4. Em **Data Access Control** (ou **API Keys**), clique em **Create** e copie a
   chave. Ela é longa, começa com `eyJ` e **só aparece uma vez**: se fechar a
   janela sem copiar, é preciso gerar outra. Essa chave vai em
   `QDRANT_API_KEY`.

**LLM (`LLM_PROVEDOR`, `LLM_API_KEY`, `LLM_MODELO`)**

O projeto não depende de um provedor específico. O mais simples é a Groq, que
tem plano gratuito:

1. Crie uma conta em https://console.groq.com.
2. Em **API Keys**, clique em **Create API Key** e copie. A chave começa com
   `gsk_` e vai em `LLM_API_KEY`.
3. Deixe `LLM_PROVEDOR=groq`.
4. Em `LLM_MODELO`, escreva o nome de um modelo que a sua conta enxerga. O
   catálogo da Groq muda com frequência: em 2026 o `llama-3.3-70b-versatile`
   saiu do ar e o padrão do projeto passou a ser `openai/gpt-oss-120b`. Se o
   nome estiver errado, o botão **Testar conexões** do chatbot lista os
   modelos disponíveis para a sua chave.

Para usar outro provedor, veja a tabela da seção **Provedor de LLM**. Com
Ollama rodando na sua máquina não é preciso chave nenhuma.

### 11.6 Carregar os documentos no Qdrant

Se a coleção `vaar_rag` ainda não existe no seu cluster, é preciso criá-la e
subir os chunks uma vez.

Abra `notebooks/03_carga_qdrant.ipynb` e execute as células de cima para baixo.
Ele lê `data/chunks/chunks_fatec_rag.jsonl`, gera os vetores denso e esparso e
sobe tudo para o Qdrant, além de salvar
`data/vocabulario_esparso.json`, que a busca lexical precisa.

> Na primeira execução ele baixa cerca de **1,2 GB** do HuggingFace, que é o
> peso do modelo de embedding. Com GPU leva poucos minutos; em CPU, bem mais.
> No Google Colab, ative a GPU em `Ambiente de execução > Alterar tipo`.

O `notebooks/02_embeddings_qdrant.ipynb` é o de demonstração da busca: serve
para conferir a recuperação depois que a carga terminou.

Este passo é feito **uma vez**. Depois disso o chatbot só consulta.

### 11.7 Rodar o chatbot

Com o ambiente virtual ativo e dentro da RAIZ:

```bash
streamlit run chatbot.py
```

O navegador abre sozinho em `http://localhost:8501`. Se não abrir, copie o
endereço que apareceu no terminal.

Para parar, volte ao terminal e aperte `Ctrl+C`.

### 11.8 Conferir se está tudo certo

Na barra lateral, os campos já vêm preenchidos com o que está no `.env`.
Clique em **Testar conexões**. O esperado são duas faixas verdes:

```
LLM: conectado a openai/gpt-oss-120b em Groq (tem plano gratuito)
Qdrant: coleção 'vaar_rag' com 2539 pontos
```

Se as duas aparecerem, pode perguntar. Se alguma vier vermelha, a mensagem diz
o que está errado; a seção 11.10 lista os casos mais comuns.

### 11.9 O que cada campo da barra lateral faz

| Campo | Para que serve |
|---|---|
| **Endpoint do cluster** | URL do Qdrant. Vem de `QDRANT_URL` |
| **API key do Qdrant** | vem de `QDRANT_API_KEY`; fica só na sessão |
| **Coleção** | nome da coleção no Qdrant, `vaar_rag` |
| **Provedor** | Anthropic, OpenAI, Groq, Gemini, Ollama ou outro compatível |
| **API key** do LLM | vem de `LLM_API_KEY`; desabilitada no Ollama, que não usa |
| **Modelo** | nome do modelo no provedor; é editável porque os catálogos mudam |
| **Modo** de busca | híbrida, só esparsa (rápida, sem Qwen) ou só densa |
| **Estágios do pipeline** | liga e desliga cada etapa; a aba Pipeline mostra as desligadas como "pulado" |
| **Candidatos da busca híbrida** | quantos trechos o banco devolve antes do reranking |
| **Trechos enviados ao LLM** | quantos sobram depois do reranking |
| **Tamanho do contexto** | teto de caracteres mandados ao LLM, para não estourar o limite do provedor |

Numa máquina sem GPU, a combinação que responde mais rápido é **Modo: só
esparsa** com o **HyDE desligado**: a busca cai de dezenas de segundos para
cerca de 50 ms. A busca híbrida acha mais coisa, mas paga o preço de vetorizar
a consulta na CPU. A seção **Onde vai o tempo** tem os números medidos.

### 11.10 Erros comuns

| Mensagem | O que aconteceu | Como resolver |
|---|---|---|
| `ModuleNotFoundError: No module named 'streamlit'` | o ambiente virtual não está ativo | ative o `.venv` (passo 11.3) |
| `endpoint do Qdrant ausente` | o campo da URL está vazio | preencha `QDRANT_URL` no `.env` ou o campo na barra lateral |
| `WinError 10061` ou `Connection refused` | a URL está vazia e o cliente tentou `localhost` | mesma coisa acima |
| `403 Forbidden` no Qdrant | chave errada, incompleta ou com espaço | gere outra chave e cole inteira, sem aspas |
| `coleção 'vaar_rag' não existe` | a carga ainda não foi feita | rode `notebooks/03_carga_qdrant.ipynb` (passo 11.6) |
| `coleção 'vaar_rag' existe mas está vazia` | a carga começou e não terminou | rode o notebook de novo até o fim |
| `API key inválida` no LLM | chave do provedor errada | confira em console.groq.com |
| `modelo 'X' não encontrado` | o nome saiu do catálogo do provedor | use um dos nomes que o **Testar conexões** lista |
| `requisição grande demais para o provedor` | o contexto passou do limite do plano | baixe **Tamanho do contexto** na barra lateral |
| `data/vocabulario_esparso.json não existe` | o arquivo do vocabulário não foi gerado | rode `notebooks/03_carga_qdrant.ipynb` |
| a busca demora mais de 30 segundos | o Qwen está vetorizando na CPU | troque o **Modo** para "só esparsa" ou desligue o **HyDE** |
| as respostas vêm com "não contém essa informação" | a busca não achou o trecho | volte para o modo híbrido, que acha mais que o esparso puro |

### 11.11 Demonstração e testes

```bash
python scripts/demo_componentes.py    # denso e lexical lado a lado, sem baixar modelo
pytest -q                             # 58 testes; usam LLM e Qdrant falsos, não gastam chave
```

Os dois rodam de dentro da RAIZ, com o ambiente virtual ativo.

---

## 12. Estrutura do repositório

```
data/
  fonte/    fundeb_vaar_atualizado.json      22 documentos extraídos
  chunks/   chunks_fatec_rag.jsonl           438 chunks, 21 documentos
  vocabulario_esparso.json                   mapa termo -> índice (versionado)

notebooks/
  02_embeddings_qdrant.ipynb                 demonstração da busca
  03_carga_qdrant.ipynb                      carga dos chunks no Qdrant

executar.bat                                 Windows: prepara tudo e abre o chatbot
executar.sh                                  Linux e macOS: o mesmo
chatbot.py                                   interface: chat, pipeline, avaliação
.env                                         chaves, criado por você, fora do Git
.streamlit/config.toml                       tema base da interface

src/
  embedding/qwen.py                          chamada do Qwen3, encapsulada
  esparso/bm25.py                            tokenização pt-BR
  interface/
    govbr.py                                 paleta, cabeçalho e ícones do gov.br
    bandeira.svg                             avatar da resposta
  pipeline/
    llm.py                                   clientes Anthropic e OpenAI-compatível
    etapas.py                                roteador, reescrita, filtros, HyDE, geração, avaliação
    recuperacao.py                           casca sobre scripts/motor_recuperacao.py + chunk pai
    orquestrador.py                          encadeia os estágios e produz o Trace
    trace.py                                 registro de por onde a pergunta passou

scripts/
  chunking.py                                etapa 3
  filtro_intencao.py                         etapa 6a
  geracao_hyde.py                            etapa 6b
  geracao_rag_final.py                       etapa 6c
  avaliacao_metricas.py                      etapa 7
  demo_componentes.py                        demonstração

docs/
  metodologia-extracao.md                    etapa 2
  chunking.md                                etapa 3
  retrieval-hibrida.md                       etapas 4 e 5
  justificativa-qwen.md                      Qwen vs E5
  justificativa-bm25.md                      BM25 vs SPLADE vs BGE-M3
  escolha-banco-vetorial.docx                comparativo dos 6 bancos
  arquitetura-alvo.md                        o que ainda não foi implementado

tests/
  test_retrieval.py                          13 testes, não baixam o modelo
```

### Sobre o vocabulário esparso

`data/vocabulario_esparso.json` **é versionado de propósito**. O Qdrant indexa o
vetor esparso por índice inteiro, não por palavra, e esse mapa mora no cliente.
A coleção na nuvem é compartilhada, o vocabulário não. Quem clonar o repositório
sem ele e tentar consultar recebe **zero resultados no lado lexical, sem erro
nenhum**.

---

## 13. Pendências conhecidas

1. **Falta a Lei nº 14.113/2020.** É a norma que institui o VAAR e é citada por
   todas as outras do corpus. Ela está no JSON de origem com `texto_completo:
   null` e não gerou chunk nenhum, porque veio de página HTML do Planalto e o
   conteúdo não foi incorporado. Perguntas sobre o texto da lei não têm resposta
   possível hoje. **É a pendência mais importante.**

2. **Sem conjunto de avaliação.** Tanto a justificativa do Qwen quanto a do BM25
   dizem explicitamente que a validação final depende de um benchmark próprio. Ele
   ainda não existe. São necessárias de 30 a 50 perguntas com o trecho correto
   anotado, para medir Recall@5 e MRR. Sem isso, ajustar `top_k` ou o peso entre
   denso e esparso é chute.

3. **"Chunk pai" é a página, não o artigo.** A recuperação hierárquica devolve
   a página de origem do sub-chunk, porque é a unidade que o chunking produziu.
   Devolver o artigo inteiro exigiria detectar a estrutura Capítulo > Artigo >
   Inciso na extração, o que ainda não existe.

4. **Sem vigência.** A arquitetura alvo prevê filtrar normas revogadas. O corpus
   não tem esse metadado, então o filtro não existe.

5. **16 chunks eram cabeçalho e rodapé** de página, com 1 a 26 tokens, coisas como
   "MINISTÉRIO DA EDUCAÇÃO" e "ANEXO". O notebook descarta na carga, mas a origem
   é o `chunking.py`, que define `MIN_CHUNK_TOKENS = 40` e mesmo assim deixou
   passar.

6. **Chunks longos demais.** O maior tem 2.399 tokens, contra a meta de 600.

7. **Sem cache de embedding da consulta.** Cada pergunta vetoriza o HyDE do
   zero. Para demonstração está bom; em uso contínuo vale cachear.

---

---

## 14. Publicar o chatbot na internet

Dá para publicar, mas o modelo de embedding é o que decide onde. Esta seção
traz os números medidos e o que fazer com eles.

### 14.1 Quanta memória o chatbot consome

Medido nesta máquina, com `psutil`, subindo um pedaço de cada vez:

| Configuração | RAM |
|---|---|
| Python vazio | 14 MB |
| + Streamlit | 46 MB |
| + qdrant-client | 93 MB |
| + vocabulário esparso (87.581 termos) | **103 MB** |
| + PyTorch | 256 MB |
| + Qwen3-Embedding-0.6B carregado | **1.717 MB** |
| + uma consulta vetorizada | 1.729 MB |

Os dois números que importam:

- **modo esparso: 103 MB.** Não carrega PyTorch nem o Qwen;
- **modo híbrido ou denso: cerca de 1,7 GB.** O Qwen tem 596 milhões de
  parâmetros em bfloat16, e são eles que ocupam a maior parte.

O reranker `bge-reranker-v2-m3`, se ligado, soma outros 2,2 GB.

### 14.2 Streamlit Community Cloud

O limite oficial é **690 MB no mínimo e 2,7 GB no máximo** de memória por
aplicativo ([documentação][limites]). Comparando com a tabela acima:

| Modo | Cabe? | Observação |
|---|---|---|
| Só esparsa | **sim, com folga** | 103 MB contra 2,7 GB |
| Híbrida ou densa | **no limite** | 1,7 GB do modelo mais o servidor; sobra pouco |
| Com reranker ligado | **não** | passa de 3,9 GB |

O modo híbrido cabe na conta, mas sem margem para imprevisto: qualquer pico
derruba o aplicativo com a mensagem "🤯 This app has gone over its resource
limits". Há ainda três incômodos, nenhum deles impeditivo:

1. **sem GPU.** Vetorizar a consulta continua custando 64 ms por token, e a
   CPU do Community Cloud costuma ser mais lenta que a de um notebook;
2. **o modelo é baixado a cada partida.** São 1,2 GB do HuggingFace. Os
   aplicativos hibernam depois de 12 horas sem visita, então a primeira
   pergunta depois de uma noite parada espera esse download;
3. **PyTorch no Linux.** O pacote padrão do PyPI vem com as bibliotecas CUDA
   junto, que não servem para nada sem GPU. Para instalar a versão só de CPU,
   acrescente no topo do `requirements.txt`:

   ```
   --extra-index-url https://download.pytorch.org/whl/cpu
   ```

**Recomendação:** se for publicar no Community Cloud, publique em modo
esparso. Basta colocar `MODO_BUSCA_PADRAO = "esparsa"` nos Secrets: o
aplicativo abre nesse modo e nunca carrega o Qwen. A busca lexical responde em
cerca de 50 ms e acha bem o que tem âncora exata ("art. 14", "Portaria 14"); o
que se perde é a busca semântica, que é justamente o que salva a pergunta
feita em linguagem de leigo.

### 14.3 Hugging Face Spaces, para rodar o modo híbrido

O plano gratuito de CPU do Spaces oferece **2 vCPU e 16 GB de RAM**, quase seis
vezes o teto do Community Cloud. O modo híbrido roda lá sem aperto, com o
mesmo código: Spaces tem um SDK Streamlit nativo, lê `requirements.txt` do
mesmo jeito e guarda as chaves em Settings > Variables and secrets, que chegam
ao processo como variáveis de ambiente, exatamente o que a função `segredo()`
do `chatbot.py` já lê.

Continua sendo CPU, então a busca híbrida segue na casa dos segundos.

### 14.4 Onde colocar as chaves em cada lugar

O `chatbot.py` lê as credenciais pela função `segredo()`, que procura primeiro
em `st.secrets` e depois nas variáveis de ambiente. Assim o mesmo código serve
aos três cenários:

| Onde roda | Onde colocar as chaves |
|---|---|
| Sua máquina | arquivo `.env` na raiz (seção 11.4) |
| Streamlit Community Cloud | Settings > Secrets, no formato TOML |
| Hugging Face Spaces | Settings > Variables and secrets |

No Community Cloud, o conteúdo dos Secrets é este, com aspas porque é TOML:

```toml
QDRANT_URL = "https://SEU-CLUSTER.sa-east-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "COLE-AQUI-A-CHAVE-DO-QDRANT"
LLM_PROVEDOR = "groq"
LLM_API_KEY = "COLE-AQUI-A-CHAVE-DO-PROVEDOR"
LLM_MODELO = "openai/gpt-oss-120b"
MODO_BUSCA_PADRAO = "esparsa"
```

Repare na diferença: no `.env` é `CHAVE=valor` sem aspas; nos Secrets é
`CHAVE = "valor"` com aspas, porque é TOML. **Não comite o `secrets.toml`**,
pelo mesmo motivo de sempre.

### 14.5 Passo a passo no Community Cloud

1. O repositório já está público em
   `github.com/ProjetoIntegrador-V/rag-vaar-nacional`, que é o que o
   Community Cloud exige.
2. Entre em https://share.streamlit.io com a conta do GitHub.
3. **Create app** > **Deploy a public app from GitHub**.
4. Preencha: repositório `ProjetoIntegrador-V/rag-vaar-nacional`, branch
   `main`, arquivo principal `chatbot.py`.
5. Em **Advanced settings**, escolha Python 3.11 ou mais novo e cole o bloco
   TOML da seção 14.4 no campo **Secrets**.
6. **Deploy**. A primeira construção demora vários minutos, quase tudo
   instalando PyTorch.

A coleção `vaar_rag` precisa já existir no Qdrant (seção 11.6). O aplicativo
publicado só consulta o banco; ele não carrega documento nenhum.

### 14.6 Resumo

| Pergunta | Resposta |
|---|---|
| Dá para subir no Streamlit Community Cloud? | Sim |
| Roda normalmente? | Em modo esparso, sim, com folga |
| O Qwen na pergunta é impedimento? | É o único. São 1,7 GB contra um teto de 2,7 GB: cabe, mas sem margem |
| Tem alternativa gratuita para o modo híbrido? | Hugging Face Spaces, com 16 GB de RAM no plano de CPU |
| Precisa mudar o código? | Não. As chaves já são lidas de `st.secrets` ou do ambiente |

[limites]: https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#resource-limits

## Base legal do corpus

EC 108/2020, Lei 14.113/2020, Lei 14.276/2021, Lei 14.711/2023, Decreto
10.656/2021, Resoluções CIF 15/2025, 17/2025 e 24/2026, Portaria Interministerial
MEC/MF 14/2025 e as notas técnicas do Inep e da SEB listadas em
`data/fonte/fundeb_vaar_atualizado.json`.
