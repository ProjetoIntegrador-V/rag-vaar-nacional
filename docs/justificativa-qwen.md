# Justificativa da escolha do Qwen3-Embedding-0.6B

Comparação com o `intfloat/multilingual-e5-large-instruct`.

> Resumo no [README](../README.md#5-etapa-4-embeddings).

---

Comparação com o intfloat/multilingual-e5-large-instruct

Os dados utilizados nesta comparação

Os valores apresentados neste documento foram retirados do MTEB Leaderboard (Massive Text Embedding Benchmark), plataforma de avaliação e comparação de modelos de embeddings. A tabela analisada foi fornecida no material deste projeto e apresenta rankings e resultados de diferentes modelos em diversos benchmarks.

O MTEB é um conjunto de benchmarks criado para avaliar modelos de representação textual em diferentes tarefas, como recuperação de informação (retrieval), similaridade semântica, classificação e reranking. Por isso, ele permite comparar modelos de embeddings em diferentes cenários, em vez de considerar apenas uma única métrica.

Nesta análise, o foco é o benchmark MTEB Portuguese, pois o sistema será utilizado principalmente com documentos em português. Os dados mostram o Qwen3-Embedding-0.6B em 2º lugar entre 240 modelos (71,63) e o multilingual-e5-large-instruct em 1º lugar (72,09).

1. Critério de escolha

A escolha do Qwen3-Embedding-0.6B não deve ser apresentada como uma afirmação de que ele é globalmente superior ao E5. O critério é a adequação ao pipeline de RAG, considerando desempenho competitivo, capacidade de entrada e flexibilidade para experimentar diferentes estratégias de chunking.

2. Desempenho em português

No benchmark MTEB para português, o E5 apresenta o melhor resultado (72,09), enquanto o Qwen ocupa a segunda posição (71,63). A diferença é de apenas 0,46 ponto. Portanto, o Qwen apresenta desempenho competitivo em português, mas não deve ser descrito como superior ao E5 nesse aspecto.

3. Principal vantagem para o pipeline: tamanho de entrada

O E5 possui limite de aproximadamente 514 tokens por entrada, enquanto o Qwen suporta até 32.768 tokens. Esse limite se aplica às entradas usadas para gerar embeddings, tanto na indexação dos chunks quanto na vetorização de uma consulta.

Para o projeto, essa diferença não significa que o Qwen deve receber o PDF inteiro. O chunking continua sendo necessário. A vantagem é oferecer maior margem para testar chunks de diferentes tamanhos sem atingir rapidamente o limite de entrada do modelo.

4. Justificativa resumida

O Qwen3-Embedding-0.6B foi selecionado por apresentar desempenho competitivo em português e, principalmente, por oferecer uma capacidade de entrada muito maior (32.768 tokens), o que proporciona maior flexibilidade para o estudo de estratégias de chunking em documentos científicos extensos. A escolha considera a adequação ao caso de uso, e não apenas a posição em um ranking geral.

5. Observação metodológica

Os rankings externos, como o MTEB, não determinam qual modelo será melhor para o corpus do projeto. A validação final deve ser feita com um benchmark próprio sobre os artigos utilizados no RAG, comparando métricas de recuperação, como Recall@k, MRR ou nDCG, além de latência e uso de memória.

Fonte dos dados

Os rankings e resultados foram retirados das páginas dos modelos no MTEB Leaderboard fornecidas no material analisado. As especificações técnicas (parâmetros, dimensão e limite de tokens) também foram apresentadas nas respectivas páginas dos modelos.

| Característica | Qwen3-Embedding-0.6B | multilingual-e5-large-instruct |
|---|---|---|
| Benchmark MTEB Português | #2 / 240 — 71,63 | #1 / 240 — 72,09 |
| Dimensão do embedding | 1.024 | 1.024 |
| Máximo de tokens | 32.768 | 514 |
| Parâmetros | 596M | 560M |
| Licença | Apache-2.0 | MIT |

| Etapa do RAG | E5 | Qwen |
|---|---|---|
| Indexação | Cada chunk deve caber no limite; textos maiores podem ser truncados. | Maior margem para chunks maiores. |
| Busca | Perguntas normalmente ficam muito abaixo do limite. | Também suporta entradas muito maiores. |
