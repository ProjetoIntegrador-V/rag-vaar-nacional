@echo off
REM ---------------------------------------------------------------------
REM  RAG VAAR Nacional - Windows
REM
REM  Clique duas vezes neste arquivo. Ele prepara o ambiente e abre o
REM  chatbot no navegador. Na primeira vez demora, porque baixa as
REM  bibliotecas; nas seguintes abre em segundos.
REM ---------------------------------------------------------------------
cd /d "%~dp0"
title RAG VAAR Nacional
echo.
echo  ===========================================================
echo    RAG VAAR Nacional
echo  ===========================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo  [ERRO] Python nao foi encontrado nesta maquina.
  echo.
  echo  Instale o Python 3.10 ou mais novo em https://www.python.org/downloads/
  echo  e marque a caixa "Add Python to PATH" durante a instalacao.
  echo.
  pause
  exit /b 1
)

if not exist ".env" (
  echo  [ERRO] O arquivo .env nao esta nesta pasta.
  echo.
  echo  Copie o arquivo .env que voce recebeu para:
  echo  %cd%
  echo.
  echo  Ele guarda as chaves do banco de dados e do modelo de linguagem.
  echo.
  pause
  exit /b 1
)

if not exist ".venv" (
  echo  [1 de 3] Criando o ambiente virtual...
  python -m venv .venv
  if errorlevel 1 goto erro
)

echo  [2 de 3] Instalando as bibliotecas. Na primeira vez demora alguns
echo           minutos e baixa cerca de 2 GB. Nao feche esta janela.
echo.
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet
if errorlevel 1 goto erro

echo.
echo  [3 de 3] Abrindo o chatbot no navegador...
echo.
echo           Endereco:  http://localhost:8501
echo           Para encerrar, feche esta janela ou aperte Ctrl+C.
echo.
start "" cmd /c "timeout /t 8 >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run chatbot.py
goto fim

:erro
echo.
echo  [ERRO] Alguma etapa falhou. A mensagem acima diz o que aconteceu.
echo.
pause
exit /b 1

:fim
pause
