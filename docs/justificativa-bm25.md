# Justificativa da escolha do BM25 como modelo esparso

Comparação com SPLADE e BGE-M3, em conjunto com o Qwen3-Embedding-0.6B.

> Resumo no [README](../README.md#6-etapa-4-modelo-esparso).

---

Comparação com SPLADE e BGE-M3 em conjunto com o Qwen-Embedding-0.6B

Contexto e Objetivo

Este documento apresenta a fundamentação técnica para a escolha do algoritmo BM25 como motor de busca esparsa (busca por palavras-chave) para compor a arquitetura de busca híbrida de um sistema RAG (Retrieval-Augmented Generation) focado em legislações do Fundeb. A análise compara o BM25 com alternativas neurais como SPLADE e BGE-M3, considerando que o pipeline já adotará o modelo Qwen-Embedding-0.6B para a geração de vetores densos (semântica) e utilizará o banco de dados vetorial Qdrant.

1. Critério de escolha

A escolha do BM25 não se baseia em complexidade, mas no princípio do custo-benefício computacional e na complementariedade de funções. Como o Qwen-Embedding-0.6B já entrega alta capacidade de compreensão semântica (vetores densos), o modelo esparso precisa apenas preencher a lacuna da busca exata por siglas e números (como valores do Fundeb), sem adicionar sobrecarga desnecessária de processamento.

2. O Problema da Redundância Computacional

Utilizar SPLADE ou BGE-M3 em conjunto com o Qwen-Embedding-0.6B gera redundância arquitetural. O SPLADE exige o carregamento de uma segunda rede neural na memória apenas para realizar expansão de vocabulário, algo que o Qwen já soluciona em sua camada semântica. Já o BGE-M3 é um modelo "All-in-One"; utilizá-lo apenas para extrair sua camada esparsa desperdiça recursos, pois a arquitetura exigiria rodar dois modelos gigantes para o mesmo texto.

3. Principal vantagem para o pipeline: Eficiência e Exatidão

O BM25 não é uma rede neural, mas uma fórmula matemática incorporada no Qdrant. Para um corpus de 20 arquivos contendo legislação educacional — onde siglas como "VAAT-MIN", "Simec", "VAAF" e quantias exatas (ex: R$ 10.194,38) não possuem sinônimos — o modelo não precisa "pensar". O BM25 age como uma lupa de precisão cirúrgica para essas âncoras lexicais, enquanto o Qwen trabalha o contexto nas entrelinhas.

4. Justificativa resumida

O BM25 foi selecionado por apresentar o melhor custo-benefício técnico. Ao terceirizar a busca exata (esparsa) para o BM25 nativo do Qdrant e a busca semântica (densa) para o Qwen-Embedding-0.6B, o sistema garante precisão absoluta em jargões e números, além de um processamento extremamente leve e barato, ideal para rodar localmente sem a necessidade de múltiplas GPUs.

5. Observação metodológica

Embora o BM25 seja teoricamente a melhor escolha para atuar ao lado do Qwen neste cenário de baixo volume (20 documentos), a eficácia final da busca híbrida dependerá da calibração do algoritmo Reciprocal Rank Fusion (RRF) no Qdrant. Os testes práticos deverão balancear o peso dado ao vetor denso e ao vetor esparso de acordo com a natureza matemática ou teórica das perguntas dos usuários.

| Característica | BM25 | SPLADE | BGE-M3 |
|---|---|---|---|
| Tipo de Modelo | Algoritmo Estatístico (TF-IDF) | Rede Neural Esparsa | Híbrido Nativo (Denso, Esparso, ColBERT) |
| Custo Computacional | Zero (Roda nativamente no banco) | Alto (Requer GPU/CPU para gerar vetores) | Muito Alto (Modelo muito pesado) |
| Redundância com Qwen | Nenhuma (Apenas exatidão) | Parcial (Expansão de vocabulário sobreposta) | Total (Tornaria o Qwen obsoleto no pipeline) |
| Espaço no Banco | Mínimo | Médio | Alto |

---

## Nota de implementação

O Qdrant fornece o **IDF**, a metade estatística do BM25, pelo
`Modifier.IDF` na configuração do vetor esparso. A **tokenização é
nossa**, em `src/esparso/bm25.py`, com regras específicas deste domínio:
remoção de acentos nos dois lados, preservação de ponto, hífen e barra
dentro do token (`art.14`, `3106200-1`, `2025/2026`) e ausência de
stemmer, que achataria justamente as âncoras exatas.

