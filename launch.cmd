@echo off
setlocal

if "%CLAUDE_OLLAMA_MODEL%"=="" set "CLAUDE_OLLAMA_MODEL=gemma4:31b"

where ollama >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Ollama is not installed.
  echo   Download: https://ollama.com
  exit /b 1
)

where claude >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Claude Code CLI is not installed.
  echo   Install: npm install -g @anthropic-ai/claude-code
  exit /b 1
)

echo =========================================
echo   Gemma 4 Local Agent
echo   Model: %CLAUDE_OLLAMA_MODEL%
echo =========================================

if "%~1"=="" (
  ollama launch claude --model %CLAUDE_OLLAMA_MODEL%
) else (
  ollama launch claude --model %CLAUDE_OLLAMA_MODEL% -- %*
)
