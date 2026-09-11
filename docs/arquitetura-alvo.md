# Arquitetura alvo da recuperação e reranking

Desenho de referência para o domínio jurídico e normativo.

> **Atenção:** este documento descreve o alvo, não o estado atual.
> Hoje estão implementados a busca híbrida e o filtro por metadado.
> Reranking, reescrita de consulta e recuperação hierárquica ainda não.
> Ver as pendências no [README](../README.md#13-pendências-conhecidas).

---

Para o domínio jurídico e normativo (leis, decretos e normas técnicas), a precisão exata é inegociável. Modelos puramente semânticos (apenas embeddings) costumam falhar nesse cenário porque lidam mal com números exatos de leis (ex: Lei nº 13.709/2018), siglas (ex: ABNT NBR ISO 9001) e numeração de artigos.

A melhor estratégia para o Tópico 5 (Recuperação e Reranking) nesse contexto exige uma abordagem multicamadas. Aqui está o desenho ideal:

1. Busca Híbrida (Hybrid Search) Obrigatória

Você não pode depender apenas de vetores densos (Embeddings). É preciso combinar duas abordagens, geralmente unidas pelo algoritmo RRF (Reciprocal Rank Fusion):

Busca Esparsa (Lexical/BM25): Fundamental para garantir o "Exact Match". Se o usuário digitar "Artigo 5º, inciso II da NR-18", o BM25 vai caçar exatamente esses termos no texto, algo que a busca vetorial costuma errar.

Busca Densa (Vetorial): Necessária para perguntas conceituais ou feitas em linguagem comum. Se o usuário perguntar "quais as regras para trabalho em altura?", o embedding entenderá que isso está semanticamente ligado aos termos técnicos da NR-35, mesmo que a pergunta não cite a norma.

2. Pré-filtragem Forte por Metadados (Metadata Filtering)

Leis mudam, são revogadas e têm hierarquia. Antes de rodar a busca no banco vetorial, você deve aplicar filtros (Hard Filters) baseados na intenção do usuário:

Vigência: Filtrar apenas leis/normas "Ativas" (evitando que o LLM responda com uma lei revogada de 1998).

Jurisdição/Órgão: Filtrar por âmbito (Federal, Estadual, Municipal) ou órgão emissor (ANVISA, ABNT, CREA).

Dica Prática: Use o próprio LLM em uma etapa inicial (Query Routing) para extrair do prompt do usuário quais metadados ele está buscando, transformando isso em um filtro (ex: WHERE status = 'vigente' AND tipo = 'decreto').

3. Hierarquia de Contexto (Small-to-Big Retrieval)

Documentos legais são altamente estruturados (Capítulo > Seção > Artigo > Parágrafo > Inciso). Se você recupera apenas o chunk de um "Inciso III", o LLM pode não saber a qual Artigo ele pertence, perdendo o sentido.

Estratégia: Indexe o banco de dados em chunks pequenos (ex: em nível de Parágrafo/Inciso) para ter alta precisão na busca. Mas, no momento da recuperação (Step 5), retorne ao LLM o chunk pai (o Artigo inteiro ou a Seção inteira).

4. Expansão e Reescrita de Query (Query Transformation)

Usuários comuns usam vocabulário leigo, enquanto os documentos usam jargão jurídico.

Implementação: Antes de fazer o embedding da pergunta do usuário, passe-a por um LLM rápido para reescrevê-la.

Exemplo: A pergunta "Qual a multa se vazar dados de clientes?" é reescrita para "Penalidades, sanções administrativas e multas pecuniárias por vazamento de dados pessoais ou infração à LGPD". A busca híbrida será feita com esta segunda frase, melhorando o Recall.

5. Reranking com Cross-Encoder

O banco vetorial (e o BM25) vão te devolver os Top-K resultados (ex: 20 documentos) de forma muito rápida, mas a ordem pode não ser perfeita para o rigor jurídico.

O que fazer: Passe esses 20 resultados por um modelo de Reranker (como Cohere Rerank ou um modelo Cross-Encoder treinado/fine-tuned em PT-BR).

O Reranker lerá a pergunta original e cada um dos 20 chunks em conjunto, pontuando a relevância exata de cada um. Isso garante que a cláusula exata que responde à pergunta suba para a posição 1. Você então envia apenas os Top-3 ou Top-5 para o prompt final do LLM (Passo 6).

Resumo do Fluxo do Tópico 5 para o seu caso: Pergunta do Usuário → Query Transformation → Extração de Metadados → Busca Híbrida (Dense + BM25) → Top 20 Chunks → Cross-Encoder Reranker → Top 5 Chunks → Substituição pelo Parent Document (Artigo completo) → Envio para o LLM.

Arquitetura Hyde – Pesquisar

https://ollama.com/qllama/bge-reranker-v2-m3

| Modelo | Hospedagem | Contexto Máximo | Ponto Forte para Documentos Legais |
|---|---|---|---|
| Cohere Rerank 3 | API | 4.096 tokens | Altíssima precisão com tabelas e formatação de normas técnicas. |
| BGE-Reranker-v2-M3 | Local / Cloud Própria | 8.192 tokens | Lida com artigos jurídicos longos sem cortar texto; open-source. |
| Multilingual MiniLM | Local / Cloud Própria | 512 tokens | Extremamente rápido e leve para testes e validações iniciais. |
| Modelos BERTimbau | Local / Cloud Própria | 512 tokens | Compreensão profunda da gramática e vocabulário PT-BR nativo. |
