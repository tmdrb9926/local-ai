# Gemma 4 Local Agent

Claude Code + Ollama Gemma 4 31B 기반 로컬 코드 에이전트.
사용하면서 자동으로 파인튜닝되어 점점 똑똑해지는 나만의 AI 코딩 어시스턴트.

## 구성

```
Claude Code (공식 CLI)  ←→  Ollama (Gemma 4 31B)  ←→  자동 파인튜닝
      │                          │                         │
  MCP, Skills,              로컬 GPU 추론             세션 로그 수집
  /commands 전부 동작        RTX 4090 24GB            → QLoRA 학습
                                                     → 모델 자동 교체
```

## 요구사항

- Windows 11
- NVIDIA RTX 4090 (24GB VRAM)
- [Ollama](https://ollama.com) 설치
- [Claude Code CLI](https://www.npmjs.com/package/@anthropic-ai/claude-code) 설치
- Python 3.12+ (파인튜닝용)

## 빠른 시작

```powershell
# 1. Gemma 4 모델 다운로드
ollama pull gemma4:31b

# 2. Claude Code + Gemma 4 실행
.\launch.cmd

# 3. (선택) 파인튜닝 환경 설정
.\scripts\setup-finetune.ps1
```

## 구조

```
gemma4-agent/
├── launch.cmd              # Claude Code + Ollama 실행기
├── launch.ps1              # PowerShell 버전
├── scripts/
│   ├── setup-finetune.ps1  # Unsloth 환경 설정
│   └── collect-data.ps1    # 세션 로그 → 학습 데이터 변환
├── finetune/
│   ├── train.py            # QLoRA 파인튜닝 스크립트
│   ├── export.py           # 세션 로그 → JSONL 변환
│   └── reload.py           # GGUF → Ollama 모델 교체
└── gateway/
    └── (Phase 3: API Gateway - 추후)
```

## 파인튜닝 흐름

```powershell
# 1. 세션 데이터 수집 (Claude Code 사용 후)
python finetune/export.py

# 2. 파인튜닝 실행
python finetune/train.py

# 3. 모델 교체
python finetune/reload.py
```
