# Recuperação: Qwen3-Embedding-0.6B + BM25

Documento de referência dos dois componentes de recuperação.

**Escopo atual:** apenas as duas chamadas, isoladas e sem persistência. A camada
de busca combinada e o banco vetorial ficam para depois, quando a equipe
decidir o banco. A seção 6 descreve como os dois se conectam, para que essa
decisão possa ser tomada com o desenho já fechado.

Implementação em `src/embedding/` (denso) e `src/esparso/` (lexical).

---

## 1. Os dois componentes

| Componente | Arquivo | Entrada | Saída |
|---|---|---|---|
| `QwenEmbedder` | `src/embedding/qwen.py` | lista de textos | matriz `(n, 1024)` de vetores normalizados |
| `BM25Retriever` | `src/esparso/bm25.py` | lista de textos + ids | `[(id, score)]` para uma consulta |

Nenhum dos dois grava em disco, abre conexão ou assume banco. São blocos puros:
mesma entrada, mesma saída.

## 2. Por que dois retrievers e não só o embedding

Os dois erram em lugares diferentes, e nesse domínio isso é decisivo.

| Consulta | Denso (Qwen3) | Lexical (BM25) |
|---|---|---|
| "quais critérios habilitam o município" | acerta (paráfrase) | erra se o documento escreve "requisitos de elegibilidade" |
| "código IBGE 3106200" | erra (o vetor borra o número) | acerta (casamento exato) |
| "art. 14 da lei do FUNDEB" | fraco | acerta |
| "portaria de 2026 sobre repasse" | acerta | parcial |

O corpus do VAAR é cheio de âncoras exatas: códigos de município, números de
portaria, artigos de lei, anos, siglas. Embedding denso sozinho perde essas
consultas de forma silenciosa, sem erro, só devolvendo o trecho errado. BM25
sozinho perde toda pergunta feita com palavras diferentes das do documento.

Isso é verificável hoje, rodando `scripts/demo_componentes.py`.

## 3. Chamada do Qwen3-Embedding-0.6B

```python
from src.embedding import QwenEmbedder

embedder = QwenEmbedder()                          # modelo carrega na 1a chamada

vetores = embedder.encode_documents(textos)        # (n, 1024), norma L2 = 1
v_query = embedder.encode_queries([pergunta])[0]   # (1024,)
```

| Característica | Valor |
|---|---|
| Parâmetros | 0.6B (28 camadas) |
| Dimensão | 1024, truncável de 32 a 1024 via MRL |
| Contexto | 32k tokens |
| Pooling | último token (EOS), não média |
| Licença | Apache 2.0 |

Três detalhes que degradam a busca sem gerar erro nenhum:

**Query e documento são codificados de forma assimétrica.** A query leva um
prefixo de instrução, o documento não:

```
Instruct: {descrição da tarefa}\nQuery:{pergunta}
```

Por isso existem `encode_queries()` e `encode_documents()` separados em vez de
um `encode()` genérico. Codificar documento com prefixo de query piora a
recuperação de forma silenciosa. A instrução fica em `embedder.DEFAULT_TASK` e
**precisa ser a mesma na geração dos vetores e na consulta**. Existe um teste
(`-m lento`) que falha se o prefixo parar de ser aplicado.

**Pooling é de último token.** Consequência: o tokenizer precisa de padding à
esquerda, senão o vetor sai do token de padding. Já configurado no wrapper.

**Vetores saem normalizados** (norma L2 = 1). Com isso, produto interno é igual
a cosseno, e a similaridade vira uma multiplicação de matriz. Quando o banco
for escolhido, configure a métrica como *cosine* ou *dot product*, tanto faz,
mas **não** como distância euclidiana.

Sobre MRL: dá para truncar para 512 ou 256 (`QwenEmbedder(dim=512)`) e
economizar armazenamento sem retreinar, custando um pouco de qualidade. Isso
vira relevante na hora de dimensionar o banco. Enquanto isso, 1024.

## 4. Chamada do BM25

```python
from src.esparso import BM25Retriever

bm25 = BM25Retriever().fit(textos, ids)
resultados = bm25.score("codigo 3106200", top_k=10)   # [(id, score)]
```

`rank_bm25` (BM25 Okapi), Python puro, em memória. Decisões de tokenização:

- **Acentos são removidos** de documento e query igualmente. Usuário de portal
  público digita "orcamento" com muita frequência, e sem isso o BM25 não casa.
- **Ponto, hífen e barra são preservados dentro do token.** Sem isso, "art.14",
  "3106200-1" e "2025/2026" seriam picados em pedaços inúteis, justamente as
  âncoras que o BM25 está aqui para acertar.
- **Sem stemmer.** RSLP aumentaria recall em texto corrido, mas achata as
  âncoras exatas. Generalização morfológica já é trabalho do retriever denso.
- **Stopwords removidas, exceto numéricas** e exceto "não", "sem", "mais" e
  "menos", que carregam significado em pergunta sobre critério de habilitação.

### Armadilha do IDF, leia antes de reportar bug

O BM25 Okapi calcula `idf(t) = log(N - df(t) + 0.5) - log(df(t) + 0.5)`, que
**vai a zero quando o termo aparece em metade dos documentos e fica negativo
acima disso**. O corte default de `score()` é `min_score=0.0`, então nesse caso
o lado lexical devolve lista vazia.

Não é bug: é o Okapi dizendo que aquele termo não discrimina nada. Mas num
corpus pequeno e especializado, palavras legítimas do domínio ("fundeb",
"educacao", "municipio") podem cair nessa faixa. Se acontecer:

- `score(..., min_score=float("-inf"))` para não cortar nada; ou
- trocar `BM25Okapi` por `BM25Plus` no `fit()`, que usa `idf = log((N+1)/df)`,
  sempre positivo, desenhado para corrigir exatamente isso.

Há um teste (`test_bm25_termo_muito_comum_zera_idf`) que fixa esse
comportamento, para a equipe reconhecer o sintoma e para flagrar a troca.

### Escala

`rank_bm25` varre o corpus inteiro por consulta. Serve bem até dezenas de
milhares de documentos, a faixa deste projeto. Se passar disso, ou se o banco
escolhido já trouxer busca textual (OpenSearch, Postgres full-text), o BM25
migra para lá e este módulo vira só o tokenizador. A assinatura de `score()`
não muda.

## 5. Como rodar

```bash
pip install -r requirements.txt
```

Demonstração das duas chamadas lado a lado (só BM25, não baixa nada):

```bash
python scripts/demo_componentes.py
```

Incluindo o Qwen (baixa cerca de 1.2 GB do HuggingFace na primeira vez):

```bash
python scripts/demo_componentes.py --com-embedding
```

Testes rápidos, sem baixar modelo:

```bash
pytest -q
```

Teste que exercita o modelo de verdade:

```bash
pytest -q -m lento
```

## 6. Como os dois vão se conectar (ainda não implementado)

Fica registrado aqui para a decisão do banco ser tomada com o desenho fechado.
Nada disso está no código ainda.

```
pergunta  ->  [ denso (Qwen3) ]   ->  50 candidatos  \
                                                      >-- RRF -->  top-k trechos
pergunta  ->  [ lexical (BM25) ]  ->  50 candidatos  /
```

**A fusão deve ser por RRF (Reciprocal Rank Fusion), não por soma ponderada
dos scores.** Motivo: o cosseno do Qwen vive num intervalo estreito e alto,
tipicamente 0.3 a 0.9, enquanto o score do BM25 é ilimitado para cima e varia
com o tamanho do corpus, a raridade dos termos e o tamanho da query. Pode dar
2 numa consulta e 40 na seguinte. Somar exige normalizar, e toda normalização
(min-max, z-score) depende do lote de resultados daquela consulta, o que faz o
peso calibrado numa pergunta não valer na outra.

RRF descarta os scores e usa só a posição:

```
RRF(d) = soma_i  peso_i / (k + rank_i(d))          k = 60
```

Livre de escala, nada para calibrar, e não quebra quando um dos retrievers
devolve poucos resultados ou nenhum (o que, pela seção 4, acontece de verdade).

São umas 20 linhas de código, sem dependência nova. O que falta para escrever
não é o algoritmo, é decidir onde os vetores ficam.

### O que a escolha do banco precisa suportar

1. **Métrica de cosseno ou produto interno** para vetores de 1024 dimensões.
2. **Devolver top-N por id**, para o RRF conseguir cruzar as duas listas. Isso
   exige que o id do chunk seja **estável entre reprocessamentos**. Se for
   gerado por posição (`chunk_0`, `chunk_1`), qualquer reindexação embaralha o
   cruzamento e a fusão quebra em silêncio.
3. **Guardar metadados junto** (`source`, `url`, `title`, `document_type`),
   senão a resposta final não tem como citar a fonte, e rastreabilidade é
   requisito do projeto.
4. Opcionalmente, **busca textual nativa**, que permitiria aposentar o
   `rank_bm25` e rodar os dois lados dentro do mesmo banco.


   anotado, para medir Recall@5. Sem isso, qualquer ajuste de peso entre denso
   e lexical é chute.
4. **Avaliar reranker** `Qwen3-Reranker-0.6B` sobre os candidatos fundidos.
   Ganho costuma ser grande, custo é uma passada extra de modelo.
