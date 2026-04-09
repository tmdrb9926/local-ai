# Unsloth 파인튜닝 환경 설정
$VenvDir = Join-Path $env:USERPROFILE ".gemma4-agent\finetune\venv"

Write-Host "========================================="
Write-Host "  Unsloth Fine-tuning Setup"
Write-Host "========================================="

# Python 확인
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "[ERROR] Python not found." -ForegroundColor Red
    Write-Host "  Install: winget install Python.Python.3.13"
    exit 1
}
Write-Host "Python: $(python --version)"

# 가상환경 생성
Write-Host "`n[1/3] Creating virtual environment..."
if (-not (Test-Path $VenvDir)) {
    python -m venv $VenvDir
    Write-Host "  Created: $VenvDir"
} else {
    Write-Host "  Already exists: $VenvDir"
}

# 활성화 + Unsloth 설치
Write-Host "[2/3] Installing Unsloth..."
& "$VenvDir\Scripts\Activate.ps1"
pip install --upgrade pip --quiet
pip install unsloth --quiet
Write-Host "  Unsloth installed"

# GPU 확인
Write-Host "[3/3] Checking GPU..."
python -c @"
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  GPU: {gpu} ({mem:.0f}GB)')
    print('  Ready for fine-tuning!')
else:
    print('  [WARN] CUDA not available!')
"@

Write-Host "`n========================================="
Write-Host "  Setup complete!"
Write-Host ""
Write-Host "  Usage:"
Write-Host "    1. Use Claude Code with Gemma 4"
Write-Host "    2. python finetune/export.py        # collect data"
Write-Host "    3. python finetune/train.py          # fine-tune"
Write-Host "    4. python finetune/reload.py         # load model"
Write-Host "========================================="
