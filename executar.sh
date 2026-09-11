#!/usr/bin/env bash
# ---------------------------------------------------------------------
#  RAG VAAR Nacional - Linux e macOS
#
#  Rode com:   bash executar.sh
# ---------------------------------------------------------------------
set -e
cd "$(dirname "$0")"

echo
echo " ==========================================================="
echo "   RAG VAAR Nacional"
echo " ==========================================================="
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo " [ERRO] Python 3 nao foi encontrado nesta maquina."
  echo "        Instale o Python 3.10 ou mais novo e tente de novo."
  exit 1
fi

if [ ! -f ".env" ]; then
  echo " [ERRO] O arquivo .env nao esta nesta pasta."
  echo
  echo "        Copie o arquivo .env que voce recebeu para:"
  echo "        $(pwd)"
  echo
  echo "        Ele guarda as chaves do banco de dados e do modelo."
  exit 1
fi

# Se o ambiente ja existe, pula a escolha e vai direto rodar.
if [ ! -x ".venv/bin/streamlit" ]; then
  echo " Primeira execucao nesta maquina. Escolha como instalar:"
  echo
  echo "   [1] RAPIDO     cerca de 3 minutos, 460 MB"
  echo "       Busca por palavra (BM25). Ja responde perguntas com"
  echo "       citacao da norma. E o suficiente para ver o sistema todo."
  echo
  echo "   [2] COMPLETO   cerca de 10 minutos, 2 GB"
  echo "       Acrescenta a busca semantica com o modelo Qwen, que entende"
  echo "       a pergunta feita em linguagem comum."
  echo
  printf " Digite 1 ou 2 (em 20 segundos segue no 1): "
  read -r -t 20 ESCOLHA || true
  echo

  if [ ! -d ".venv" ]; then
    echo " [1 de 3] Criando o ambiente virtual..."
    python3 -m venv .venv
  fi

  if [ "$ESCOLHA" = "2" ]; then
    echo " [2 de 3] Instalando tudo, inclusive o PyTorch. Sao cerca de 2 GB."
    echo "          As linhas abaixo mostram o progresso."
    echo
    .venv/bin/python -m pip install -r requirements.txt
  else
    echo " [2 de 3] Instalando o essencial. Sao cerca de 460 MB."
    echo "          As linhas abaixo mostram o progresso."
    echo
    .venv/bin/python -m pip install -r requirements-minimo.txt
    # Sem o PyTorch nao ha vetor denso: o app abre na busca esparsa.
    echo "MODO_BUSCA_PADRAO=esparsa" > .modo_instalado
  fi
fi

# A instalacao rapida nao tem o modelo de embedding; abre em modo esparso.
if [ -f ".modo_instalado" ]; then
  export MODO_BUSCA_PADRAO=esparsa
fi

echo
echo " [3 de 3] Abrindo o chatbot..."
echo
echo "          Endereco:  http://localhost:8501"
echo "          Para encerrar, aperte Ctrl+C."
echo
( sleep 8; (xdg-open http://localhost:8501 || open http://localhost:8501) >/dev/null 2>&1 ) &
.venv/bin/python -m streamlit run chatbot.py
