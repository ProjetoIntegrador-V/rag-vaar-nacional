# RAG VAAR Nacional

Sistema de perguntas e respostas sobre a legislação do **Fundeb** e da
**Complementação da União por resultados (VAAR)**, com respostas ancoradas em
normas oficiais e citação obrigatória da fonte.

Projeto Integrador V.

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
11. [Como rodar](#11-como-rodar)
12. [Estrutura do repositório](#12-estrutura-do-repositório)
13. [Pendências conhecidas](#13-pendências-conhecidas)

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
[6a] Filtro de intenção ......... fora de escopo? o pipeline para aqui
   |
   v
[6b] HyDE ....................... LLM escreve um documento hipotético
   |
   v
[4]  Embedding da consulta ...... Qwen3-Embedding-0.6B (denso)
     Tokenização da consulta .... pt-BR (esparso)
   |
   v
[5]  Busca no Qdrant ............ denso + esparso em paralelo
     Fusão por RRF .............. nativa do banco
   |
   v
[6c] Geração ancorada ........... resposta com citação da fonte
   |
   v
[7]  Avaliação .................. score de factualidade
```

| Etapa | O que faz | Onde está | Estado |
|---|---|---|---|
| 1. Coleta | Reúne as normas | `data/fonte/` | pronto |
| 2. Extração | PDF e HTML para JSON | `data/fonte/fundeb_vaar_atualizado.json` | pronto, 1 falha |
| 3. Chunking | Segmenta em ~600 tokens | `scripts/chunking.py` | pronto |
| 4. Embedding e indexação | Vetoriza e sobe ao Qdrant | `notebooks/02_embeddings_qdrant.ipynb` | pronto |
| 5. Recuperação | Busca híbrida com RRF | mesmo notebook, função `buscar()` | pronto |
| 6. Geração | Roteamento, HyDE e resposta | `scripts/geracao_*.py` | esqueleto |
| 7. Avaliação | Factualidade | `scripts/avaliacao_metricas.py` | esqueleto |

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

## 11. Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Credenciais

Copie `.env.example` para `.env` e preencha:

```
QDRANT_URL=https://SEU-CLUSTER.sa-east-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=sua-chave
```

O `.env` está no `.gitignore`. **Nunca comite a chave**: uma vez no histórico do
Git ela vaza, mesmo que o arquivo seja apagado depois. Se preferir não criar
arquivo, o notebook pergunta na hora e a chave não aparece na tela.

### Etapa 4, embeddings e carga

Abra `notebooks/02_embeddings_qdrant.ipynb` e execute de cima para baixo. Ele
cria a coleção, sobe os 422 chunks úteis e valida a busca.

> Na primeira execução baixa cerca de **1,2 GB** do HuggingFace, o peso do
> modelo. Com GPU leva poucos minutos; em CPU, bem mais. No Colab, ative a GPU em
> `Ambiente de execução > Alterar tipo`.

Para rodar sem conta e sem rede, troque `MODO_QDRANT` para `"local"` na célula de
configuração. O resto do notebook é idêntico.

### Demonstração e testes

```bash
python scripts/demo_componentes.py    # denso e lexical lado a lado, sem baixar modelo
pytest -q                             # 13 testes
```

---

## 12. Estrutura do repositório

```
data/
  fonte/    fundeb_vaar_atualizado.json      22 documentos extraídos
  chunks/   chunks_fatec_rag.jsonl           438 chunks, 21 documentos
  vocabulario_esparso.json                   mapa termo -> índice (versionado)

notebooks/
  02_embeddings_qdrant.ipynb                 etapas 4 e 5

src/
  embedding/qwen.py                          chamada do Qwen3, encapsulada
  esparso/bm25.py                            tokenização pt-BR

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

3. **Sem reranking.** A arquitetura desenhada em `docs/arquitetura-alvo.md`
   prevê um cross-encoder reordenando os 20 primeiros resultados antes de enviar
   os 5 melhores ao LLM. Não foi implementado.

4. **Sem recuperação hierárquica.** A mesma arquitetura prevê indexar em chunks
   pequenos e devolver o artigo inteiro ao LLM. Hoje o chunk recuperado é o chunk
   enviado.

5. **16 chunks eram cabeçalho e rodapé** de página, com 1 a 26 tokens, coisas como
   "MINISTÉRIO DA EDUCAÇÃO" e "ANEXO". O notebook descarta na carga, mas a origem
   é o `chunking.py`, que define `MIN_CHUNK_TOKENS = 40` e mesmo assim deixou
   passar.

6. **Chunks longos demais.** O maior tem 2.399 tokens, contra a meta de 600.

7. **Os scripts de geração são esqueletos.** Recebem `cliente_llm` como parâmetro,
   mas nenhum cliente está implementado, e não há tratamento de erro.

---

## Base legal do corpus

EC 108/2020, Lei 14.113/2020, Lei 14.276/2021, Lei 14.711/2023, Decreto
10.656/2021, Resoluções CIF 15/2025, 17/2025 e 24/2026, Portaria Interministerial
MEC/MF 14/2025 e as notas técnicas do Inep e da SEB listadas em
`data/fonte/fundeb_vaar_atualizado.json`.
