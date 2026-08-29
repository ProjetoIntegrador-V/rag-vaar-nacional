RAG VAAR Nacional

Repositório responsável pela coleta, organização, tratamento e preparação de informações utilizadas pelo sistema de RAG (Retrieval-Augmented Generation) do VAAR Nacional.

O projeto tem como objetivo centralizar e estruturar diferentes fontes de informação, preparando os dados para posterior indexação, recuperação semântica e utilização por aplicações baseadas em Inteligência Artificial.

Objetivo

O RAG VAAR Nacional busca fornecer uma base de informações confiável e organizada para apoiar a recuperação de conteúdos relevantes durante a interação com modelos de linguagem.

Este repositório concentra principalmente as etapas relacionadas à aquisição e preparação dos dados, permitindo que as informações sejam transformadas em um formato adequado para os processos posteriores de busca e geração de respostas.

Fluxo de dados

De forma geral, o pipeline segue as seguintes etapas:

Fontes de informação
        ↓
Coleta
        ↓
Extração
        ↓
Limpeza e tratamento
        ↓
Normalização
        ↓
Estruturação dos documentos
        ↓
Chunking / Segmentação
        ↓
Geração de embeddings
        ↓
Indexação
        ↓
Recuperação semântica
        ↓
RAG VAAR Nacional

Fontes de informação

As informações podem ser obtidas a partir de diferentes fontes, de acordo com os dados necessários ao projeto, como:

Documentos e arquivos institucionais;
Bases de dados;
APIs e serviços externos;
Páginas e conteúdos públicos;
Dados estruturados e não estruturados;
Outras fontes relevantes para o VAAR Nacional.

Cada fonte pode possuir um processo específico de coleta e tratamento.

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
├── scripts/              # Scripts auxiliares e de execução
├── tests/                # Testes automatizados
├── docs/                 # Documentação adicional
├── requirements.txt      # Dependências Python
└── README.md


A estrutura acima é uma referência e deve ser adaptada conforme a implementação atual do projeto.

Tecnologias

As tecnologias utilizadas podem variar de acordo com cada etapa do pipeline. Entre os principais componentes estão:

Python — desenvolvimento dos pipelines e rotinas de processamento;
RAG — estratégia de recuperação e geração de informações;
Embeddings — representação semântica dos conteúdos;
Vector Store — armazenamento e recuperação por similaridade;
APIs — integração com fontes externas;
Docker — padronização do ambiente de execução, quando aplicável.
Instalação

Clone o repositório:

git clone <URL_DO_REPOSITORIO>
cd rag-vaar-nacional


Crie um ambiente virtual:

python -m venv .venv


Linux/macOS:

source .venv/bin/activate


Windows:

.venv\Scripts\activate


Instale as dependências:

pip install -r requirements.txt

Configuração

As variáveis de ambiente e credenciais utilizadas pelo projeto devem ser configuradas localmente.

Crie um arquivo .env:

API_KEY=
DATABASE_URL=
VECTOR_STORE_URL=
EMBEDDING_MODEL=


Nunca versione chaves de API, credenciais, tokens ou outras informações sensíveis no repositório.

Execução

Os pipelines de coleta e processamento podem ser executados por meio dos scripts disponibilizados no projeto.

Exemplo:

python scripts/collect.py


Processamento:

python scripts/process.py


Os comandos acima são exemplos e devem ser ajustados de acordo com os scripts efetivamente disponíveis no projeto.

Pipeline de processamento
1. Coleta

Os dados são obtidos a partir das fontes configuradas no projeto.

Nesta etapa são preservadas, sempre que possível, informações de origem e metadados importantes para rastreabilidade.

2. Extração

Os conteúdos relevantes são extraídos das fontes coletadas, convertendo diferentes formatos para uma estrutura comum.

3. Tratamento

Os dados passam por processos como:

Remoção de conteúdo irrelevante;
Normalização de textos;
Padronização de campos;
Tratamento de caracteres;
Deduplicação;
Validação dos dados;
Preservação de metadados.
4. Segmentação

Documentos extensos podem ser divididos em partes menores (chunks), facilitando a recuperação dos trechos mais relevantes durante uma consulta.

5. Embeddings

Os conteúdos processados podem ser convertidos em representações vetoriais (embeddings), permitindo buscas baseadas em similaridade semântica.

6. Indexação

Os documentos e seus respectivos vetores são disponibilizados para o mecanismo de
