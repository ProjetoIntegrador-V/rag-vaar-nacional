@echo off
REM ---------------------------------------------------------------------
REM  RAG VAAR Nacional - Windows
REM
REM  Clique duas vezes neste arquivo. Ele prepara o ambiente e abre o
REM  chatbot no navegador.
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

REM  Se o ambiente ja existe, pula a escolha e vai direto rodar.
if exist ".venv\Scripts\streamlit.exe" goto rodar

echo  Primeira execucao nesta maquina. Escolha como instalar:
echo.
echo    [1] RAPIDO     cerca de 3 minutos, 460 MB
echo        Busca por palavra (BM25). Ja responde perguntas com
echo        citacao da norma. E o suficiente para ver o sistema todo.
echo.
echo    [2] COMPLETO   cerca de 10 minutos, 2 GB
echo        Acrescenta a busca semantica com o modelo Qwen, que entende
echo        a pergunta feita em linguagem comum.
echo.
choice /c 12 /t 20 /d 1 /m " Digite 1 ou 2 (em 20 segundos segue no 1)"
if errorlevel 2 (set MODO=completo) else (set MODO=rapido)
echo.

if not exist ".venv" (
  echo  [1 de 3] Criando o ambiente virtual...
  python -m venv .venv
  if errorlevel 1 goto erro
)

if "%MODO%"=="completo" (
  echo  [2 de 3] Instalando tudo, inclusive o PyTorch. Sao cerca de 2 GB.
  echo           Nao feche esta janela. As linhas abaixo mostram o progresso.
  echo.
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
) else (
  echo  [2 de 3] Instalando o essencial. Sao cerca de 460 MB.
  echo           Nao feche esta janela. As linhas abaixo mostram o progresso.
  echo.
  ".venv\Scripts\python.exe" -m pip install -r requirements-minimo.txt
  REM  Sem o PyTorch nao ha vetor denso: o app abre na busca esparsa.
  echo MODO_BUSCA_PADRAO=esparsa> .modo_instalado
)
if errorlevel 1 goto erro

:rodar
REM  A instalacao rapida nao tem o modelo de embedding; abre em modo esparso.
if exist ".modo_instalado" set MODO_BUSCA_PADRAO=esparsa

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
