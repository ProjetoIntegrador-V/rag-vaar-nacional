# RAG VAAR Nacional

Sistema de perguntas e respostas sobre a legislação do **Fundeb** e da
**Complementação da União por resultados (VAAR)**, com respostas ancoradas em
normas oficiais e citação da fonte.

Projeto Integrador V.

---

## Pipeline

| Etapa | O que faz | Onde está | Estado |
|---|---|---|---|
| 1. Coleta | Reúne normas, notas técnicas e resoluções | `data/fonte/` | pronto |
| 2. Extração | PDF e HTML para JSON estruturado | `data/fonte/fundeb_vaar_atualizado.json` | pronto, com 1 falha |
| 3. Chunking | Segmenta em trechos de ~600 tokens | `scripts/chunking.py` | pronto |
| 4. Embeddings e indexação | Vetoriza e sobe para o Qdrant | `notebooks/02_embeddings_qdrant.ipynb` | pronto |
| 5. Recuperação | Busca híbrida densa + lexical com RRF | `src/`, notebook etapa 4 | pronto |
| 6. Geração | Roteamento, HyDE e resposta final | `scripts/geracao_*.py` | esqueleto |
| 7. Avaliação | Factualidade da resposta | `scripts/avaliacao_metricas.py` | esqueleto |

## Como funciona a recuperação

Dois retrievers, porque eles erram em lugares diferentes:

| Consulta | Denso (Qwen3) | Lexical (BM25) |
|---|---|---|
| "quais critérios habilitam o município" | acerta, entende a paráfrase | erra se a norma escreve "requisitos de elegibilidade" |
| "art. 14 da Lei 14.113" | fraco, o vetor borra o número | acerta, casamento exato |
| "Portaria 14/2025" | fraco | acerta |

O corpus é cheio de âncoras exatas: artigos de lei, números de portaria, siglas,
anos. Denso sozinho perde essas consultas **em silêncio**, devolvendo o trecho
errado sem erro nenhum.

Os dois resultados são fundidos por **RRF (Reciprocal Rank Fusion)**, nativo do
Qdrant. RRF usa só a posição, não o score, porque o cosseno do Qwen vive entre
0,3 e 0,9 enquanto o score lexical é ilimitado e varia com o tamanho da consulta.
Somar os dois exigiria uma normalização que muda a cada pergunta.

Detalhamento em [`docs/retrieval-hibrida.md`](docs/retrieval-hibrida.md).

## Estrutura

```
data/
  fonte/    fundeb_vaar_atualizado.json    22 documentos extraídos
  chunks/   chunks_fatec_rag.jsonl         438 chunks, 21 documentos
  qdrant/                                   coleção local (não versionada)

notebooks/
  02_embeddings_qdrant.ipynb    etapa 4: embeddings + carga no Qdrant

src/
  embedding/qwen.py             Qwen3-Embedding-0.6B, chamada encapsulada
  esparso/bm25.py               tokenização pt-BR e BM25 de referência

scripts/
  chunking.py                   etapa 3
  filtro_intencao.py            roteador de intenção
  geracao_hyde.py               documento hipotético (HyDE)
  geracao_rag_final.py          resposta final com citação
  avaliacao_metricas.py         score de factualidade
  demo_componentes.py           demonstra denso e lexical lado a lado

docs/
  retrieval-hibrida.md          decisões da camada de recuperação
  chunking.md                   estratégia de segmentação
  escolha-banco-vetorial.docx   comparativo que levou ao Qdrant

tests/
  test_retrieval.py             13 testes, não baixam o modelo
```

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Etapa 4, embeddings e carga no Qdrant.** Abra
`notebooks/02_embeddings_qdrant.ipynb` e execute de cima para baixo. Ele cria a
coleção `vaar_rag`, sobe os 422 chunks úteis e valida a busca.

> Na primeira execução baixa cerca de **1,2 GB** do HuggingFace, o peso do
> modelo. Com GPU leva poucos minutos; em CPU, bem mais. No Colab, ative a GPU
> em `Ambiente de execução > Alterar tipo`.

Demonstração dos dois retrievers sem baixar modelo:

```bash
python scripts/demo_componentes.py
```

Testes rápidos:

```bash
pytest -q
```

## Contrato entre as etapas

A função `buscar()` do notebook devolve exatamente o formato que o
`scripts/geracao_rag_final.py` já consome:

```python
[{"titulo": ..., "ano": ..., "texto": ..., "page": ..., "tipo_documento": ...}, ...]
```

Então o encaixe é direto, sem ninguém alterar código:

```python
rota = roteador_de_intencao(pergunta, cliente_llm)
if rota["status"] == "aprovado":
    docs = buscar(rota["pergunta"], top_k=5)
    resposta = gerar_resposta_final(rota["pergunta"], docs, cliente_llm)
```

Para o HyDE, vetorize o documento hipotético em vez da pergunta crua:

```python
hipotetico = gerar_documento_hyde(pergunta, cliente_llm)
docs = buscar(hipotetico, top_k=5)
```

## Metadados disponíveis para filtro

Cada chunk carrega `ano`, `tipo_documento`, `orgao`, `document_id`, `page` e
`chunk_id`. O `buscar()` aceita filtro por ano e por tipo de documento, útil para
perguntas do tipo "o que mudou nas regras de 2026".

## Decisões técnicas que valem conhecer

**Qwen3-Embedding-0.6B, e a assimetria.** A query leva um prefixo de instrução, o
documento não. Errar isso não gera exceção, só piora a recuperação. Por isso
existem `encode_queries()` e `encode_documents()` separados, e a instrução fica
fixa em um só lugar.

**O IDF fica no servidor.** A coleção usa `Modifier.IDF`, então o cliente envia
só a frequência do termo. Isso evita a armadilha do BM25 Okapi, cujo IDF zera
quando o termo aparece em metade dos documentos, o que derrubaria palavras
legítimas do domínio como "fundeb" e "municipio".

**Ids determinísticos.** Cada ponto tem um UUID5 derivado do `chunk_id`, então
rodar o notebook de novo atualiza os mesmos pontos em vez de duplicar a base.

**O filtro vai dentro de cada `prefetch`.** Numa consulta com fusão, o filtro
passado só no topo não propaga para as sub-consultas: cada lado traz candidatos
sem filtro e o RRF apenas reordena. O resultado vaza documentos de outros anos.
Isso foi verificado na prática e está comentado no notebook.

## Pendências conhecidas

1. **Falta a Lei nº 14.113/2020.** É a norma que institui o VAAR e é citada por
   todas as outras do corpus. Ela está no JSON de origem com
   `texto_completo: null` e não gerou chunk nenhum. Perguntas sobre o texto da
   lei não têm resposta possível hoje. É a pendência mais importante.
2. **16 chunks eram cabeçalho e rodapé** de página, com 1 a 26 tokens, coisas
   como "MINISTÉRIO DA EDUCAÇÃO" e "ANEXO". O notebook descarta na carga, mas a
   origem é o `chunking.py`, que define `MIN_CHUNK_TOKENS = 40` e mesmo assim
   deixou passar.
3. **Chunks longos demais.** O maior tem 2.399 tokens, contra a meta de 600. Um
   chunk grande dilui o vetor e piora a precisão da recuperação.
4. **Sem conjunto de avaliação.** O `avaliacao_metricas.py` mede factualidade da
   geração, não a qualidade da recuperação. Antes de ajustar `top_k` ou pesos,
   monte de 30 a 50 perguntas com o trecho correto anotado e meça Recall@5. Sem
   isso, qualquer ajuste é chute.
5. **Os scripts de geração são esqueletos.** Recebem `cliente_llm` como parâmetro
   mas nenhum cliente está implementado, e não há tratamento de erro.

## Base legal do corpus

EC 108/2020, Lei 14.113/2020, Lei 14.276/2021, Lei 14.711/2023,
Decreto 10.656/2021, Resoluções CIF 15/2025, 17/2025 e 24/2026, Portaria
Interministerial MEC/MF 14/2025 e as notas técnicas do Inep e da SEB listadas em
`data/fonte/fundeb_vaar_atualizado.json`.
