#!/usr/bin/env bash
# ---------------------------------------------------------------------
#  RAG VAAR Nacional - Linux e macOS
#
#  Rode com:   bash executar.sh
#
#  Ele prepara o ambiente e abre o chatbot no navegador. Na primeira vez
#  demora, porque baixa as bibliotecas; nas seguintes abre em segundos.
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

if [ ! -d ".venv" ]; then
  echo " [1 de 3] Criando o ambiente virtual..."
  python3 -m venv .venv
fi

echo " [2 de 3] Instalando as bibliotecas. Na primeira vez demora alguns"
echo "          minutos e baixa cerca de 2 GB."
echo
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -r requirements.txt --quiet

echo
echo " [3 de 3] Abrindo o chatbot..."
echo
echo "          Endereco:  http://localhost:8501"
echo "          Para encerrar, aperte Ctrl+C."
echo
( sleep 8; (xdg-open http://localhost:8501 || open http://localhost:8501) >/dev/null 2>&1 ) &
.venv/bin/python -m streamlit run chatbot.py
