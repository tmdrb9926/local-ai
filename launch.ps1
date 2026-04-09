param(
    [Parameter(ValueFromRemainingArguments)]
    [string[]]$ExtraArgs
)

$model = if ($env:CLAUDE_OLLAMA_MODEL) { $env:CLAUDE_OLLAMA_MODEL } else { "gemma4:31b" }

# Check prerequisites
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Ollama is not installed." -ForegroundColor Red
    Write-Host "  Download: https://ollama.com"
    exit 1
}
if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Claude Code CLI is not installed." -ForegroundColor Red
    Write-Host "  Install: npm install -g @anthropic-ai/claude-code"
    exit 1
}

# Check model exists
$tags = ollama list 2>$null | Select-String $model
if (-not $tags) {
    Write-Host "[WARN] Model '$model' not found. Pulling..." -ForegroundColor Yellow
    ollama pull $model
}

Write-Host "========================================="
Write-Host "  Gemma 4 Local Agent"
Write-Host "  Model: $model"
Write-Host "========================================="

if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
    ollama launch claude --model $model -- @ExtraArgs
} else {
    ollama launch claude --model $model
}
