RAG VAAR Nacional

Repositório responsável pela coleta, organização, tratamento e preparação de informações utilizadas pelo sistema de RAG (Retrieval-Augmented Generation) do VAAR Nacional.

O projeto centraliza diferentes fontes de informação e prepara os dados para processos de indexação, recuperação semântica e utilização por aplicações baseadas em Inteligência Artificial.

Objetivo

O projeto tem como objetivo construir e manter uma base de informações estruturada e confiável para utilização na arquitetura RAG do VAAR Nacional.

O pipeline contempla etapas de coleta, extração, tratamento, normalização e preparação dos dados, garantindo que os conteúdos estejam adequados para posterior recuperação e geração de respostas.

Fluxo de dados
Fontes de informação
        |
        v
     Coleta
        |
        v
     Extração
        |
        v
Limpeza e tratamento
        |
        v
    Normalização
        |
        v
   Estruturação
        |
        v
Chunking / Segmentação
        |
        v
Geração de embeddings
        |
        v
    Indexação
        |
        v
Recuperação semântica
        |
        v
   RAG VAAR Nacional

Fontes de informação

O projeto pode trabalhar com diferentes tipos de fontes, incluindo:

Documentos institucionais;
Arquivos estruturados e não estruturados;
Bases de dados;
APIs e serviços externos;
Páginas e conteúdos públicos;
Outras fontes relevantes para o VAAR Nacional.

Cada fonte pode possuir um processo específico de coleta, extração e tratamento.

Estrutura do projeto
.
├── data/
│   ├── raw/              # Dados brutos coletados
│   ├── processed/        # Dados tratados
│   └── output/           # Dados preparados para ingestão
│
├── src/
│   ├── collectors/       # Rotinas de coleta
│   ├── extractors/       # Extração de conteúdo
│   ├── processors/       # Limpeza e tratamento
│   ├── chunkers/         # Segmentação dos documentos
│   └── pipelines/        # Pipelines de processamento
│
├── scripts/              # Scripts auxiliares
├── tests/                # Testes automatizados
├── docs/                 # Documentação
├── requirements.txt      # Dependências do projeto
└── README.md


A estrutura acima é uma referência e deve ser ajustada conforme a implementação atual do projeto.

Pipeline
Coleta

Os dados são obtidos a partir das fontes configuradas no projeto.

Sempre que possível, são preservadas informações sobre a origem dos dados e seus respectivos metadados para garantir rastreabilidade.

Extração

Os conteúdos relevantes são extraídos das fontes coletadas e convertidos para uma estrutura comum de processamento.

Tratamento

Os dados passam por processos de:

Limpeza;
Normalização;
Padronização;
Remoção de conteúdos irrelevantes;
Deduplicação;
Validação;
Tratamento de caracteres;
Organização dos metadados.
Segmentação

Documentos extensos podem ser divididos em partes menores, chamadas de chunks.

A segmentação facilita a recuperação de trechos relevantes durante as consultas realizadas pelo sistema RAG.

Embeddings

Os conteúdos processados podem ser convertidos em representações vetoriais por meio de modelos de embeddings.

Essas representações permitem realizar buscas baseadas em similaridade semântica.

Indexação

Após o processamento, os documentos e seus respectivos embeddings são preparados para indexação no mecanismo de armazenamento utilizado pela arquitetura RAG.

Metadados

Sempre que possível, os documentos devem manter informações que permitam identificar sua origem e contexto.

Exemplo:

{
  "source": "fonte",
  "url": "https://exemplo.gov.br",
  "title": "Título do documento",
  "document_type": "documento",
  "created_at": "2026-01-01",
  "updated_at": "2026-01-01"
}


A estrutura definitiva dos metadados deve seguir o padrão definido pela implementação do projeto.

Configuração

As configurações específicas do ambiente devem ser armazenadas por meio de variáveis de ambiente.

Exemplo de arquivo .env:

API_KEY=
DATABASE_URL=
VECTOR_STORE_URL=
EMBEDDING_MODEL=


Credenciais, tokens e chaves de API não devem
