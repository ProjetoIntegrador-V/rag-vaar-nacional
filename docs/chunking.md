# RAG FATEC Cotia — Chunking / Segmentação

Implementação da etapa **3 — Chunking / Segmentação** de um pipeline RAG.

## Objetivo

A estratégia foi pensada para:

- baixo custo;
- fácil reprodução;
- execução local;
- preservação do contexto;
- integração posterior com embeddings e banco vetorial.

## Estratégia

O código utiliza:

- divisão preferencial por parágrafos;
- chunks de aproximadamente **600 tokens**;
- overlap de aproximadamente **15% (90 tokens)**;
- preservação do número da página;
- metadados por chunk;
- saída em formato JSONL.

O valor de 600 tokens é um ponto inicial dentro da faixa de 500–800 tokens. Outros valores podem ser testados para comparação experimental.

## Estrutura

```text
rag-chunking-fatec/
├── data/
│   └── pdfs/
├── output/
├── src/
│   └── chunking.py
├── requirements.txt
└── README.md
```

## Como executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Adicionar os PDFs

Coloque os documentos em:

```text
data/pdfs/
```

### 3. Executar

```bash
python src/chunking.py
```

### 4. Resultado

Os chunks serão salvos em:

```text
output/chunks.jsonl
```

Cada linha representa um chunk e contém metadados como:

```json
{
  "document_id": "documento",
  "title": "documento",
  "source_file": "documento.pdf",
  "page": 1,
  "section": null,
  "subsection": null,
  "chunk_id": "documento_p1_c0",
  "chunk_index": 0,
  "token_count": 587,
  "text": "..."
}
```

## Justificativa

A utilização de chunking estrutural baseado em parágrafos, combinada com um limite fixo de tokens e overlap, evita a necessidade de utilizar um LLM adicional para segmentar cada documento.

Isso reduz custo e complexidade e torna o processo mais reprodutível, pois todos os documentos podem ser processados utilizando os mesmos parâmetros.

O chunking semântico pode ser implementado posteriormente como uma abordagem experimental para comparar seu impacto na recuperação.

## Próximos testes

Para avaliar o impacto do chunking, podem ser comparadas configurações como:

| Configuração | Chunk | Overlap |
|---|---:|---:|
| Pequena | 300 | 50 |
| Média | 600 | 90 |
| Grande | 800 | 100 |

A melhor configuração deve ser definida a partir das métricas de recuperação do RAG.
