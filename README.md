# Local AI Agent

Claude Code + Ollama Gemma 4 31B 기반 로컬 코드 에이전트.
Anthropic API 키 없이, 로컬 GPU로 Claude Code의 모든 기능을 사용합니다.
사용하면서 자동으로 파인튜닝되어 점점 똑똑해지는 나만의 AI 코딩 어시스턴트.

## 시스템 아키텍처

```
┌─ 서버 PC (Windows, RTX 4090) ──────────────────────────────────┐
│                                                                 │
│  ┌───────────────┐    ┌──────────────┐    ┌──────────────────┐ │
│  │ Claude Code    │───▶│ Ollama       │    │ API Gateway      │ │
│  │ (공식 CLI)     │    │ Gemma 4 31B  │◀───│ :9000            │ │
│  │               │    │ Q4_K_M 20GB  │    │ 인증+Rate Limit  │ │
│  └───────────────┘    └──────────────┘    └────────┬─────────┘ │
│         │                     │                     │           │
│  MCP, Skills, Hooks      GPU 추론              외부 접근        │
│  /commands 전부 동작      RTX 4090 24GB                         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 자동 파인튜닝 파이프라인                                    │  │
│  │ 세션 로그 수집 → JSONL 변환 → QLoRA 학습 → 모델 교체       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         ▲                                        ▲
         │ localhost                               │ 네트워크
         │                                        │
┌─ 서버 PC 로컬 사용 ──┐              ┌─ 외부 PC (Mac/Win/Linux) ─┐
│                      │              │                           │
│ .\launch.cmd         │              │ OLLAMA_HOST=서버IP:9000   │
│ 또는                  │              │ ollama launch claude     │
│ ollama launch claude │              │   --model gemma4:31b     │
└──────────────────────┘              └───────────────────────────┘
```

## 핵심 특징

- **Claude Code 그대로**: MCP 서버, Skills, Plugins, /commands, 탭 자동완성, 마크다운 렌더링 전부 동작
- **완전 로컬**: Anthropic API 키 불필요, 인터넷 없이 사용 가능
- **외부 공유**: 같은 네트워크의 다른 PC에서 접속 가능 (API Gateway)
- **자동 파인튜닝**: 사용하면서 데이터 수집 → QLoRA 학습 → 모델 자동 교체

---

## 1. 서버 PC 설정 (Windows, NVIDIA GPU)

### 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| OS | Windows 10/11 | Windows 11 |
| GPU | NVIDIA 8GB VRAM | RTX 4090 24GB |
| RAM | 16GB | 32GB |
| 디스크 | 30GB 여유 | 100GB+ (파인튜닝 시) |

### 설치

```powershell
# 1. Ollama 설치
winget install Ollama.Ollama

# 2. Claude Code CLI 설치 (Node.js 필요)
npm install -g @anthropic-ai/claude-code

# 3. Gemma 4 모델 다운로드 (~20GB)
ollama pull gemma4:31b

# 4. 프로젝트 클론
git clone https://github.com/tmdrb9926/local-ai.git
cd local-ai
```

### 로컬 실행

```powershell
# 방법 1: 런처 사용
.\launch.cmd

# 방법 2: 직접 실행
ollama launch claude --model gemma4:31b

# 방법 3: 다른 모델 사용
$env:CLAUDE_OLLAMA_MODEL = "qwen3-30b"
.\launch.cmd
```

### 외부 접근용 게이트웨이 실행

```powershell
# 의존성 설치 (최초 1회)
pip install aiohttp

# API 키 생성
python gateway/server.py key create "팀원A"
# → gw-sk-xxxxx 출력됨 (저장해둘 것)

# 게이트웨이 시작
python gateway/server.py serve --port 9000
```

---

## 2. 외부 PC 설정

### Mac

```bash
# 1. Ollama 설치
brew install ollama

# 2. Claude Code 설치
npm install -g @anthropic-ai/claude-code

# 3. 서버 연결 설정 (영구)
echo 'export OLLAMA_HOST="http://서버IP:9000"' >> ~/.zshrc
source ~/.zshrc

# 4. 실행
ollama launch claude --model gemma4:31b
```

### Windows

```powershell
# 1. Ollama 설치
winget install Ollama.Ollama

# 2. Claude Code 설치
npm install -g @anthropic-ai/claude-code

# 3. 서버 연결 설정 (영구)
[Environment]::SetEnvironmentVariable('OLLAMA_HOST', 'http://서버IP:9000', 'User')

# 4. 새 터미널 열고 실행
ollama launch claude --model gemma4:31b
```

### Linux

```bash
# 1. Ollama 설치
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Claude Code 설치
npm install -g @anthropic-ai/claude-code

# 3. 서버 연결 설정 (영구)
echo 'export OLLAMA_HOST="http://서버IP:9000"' >> ~/.bashrc
source ~/.bashrc

# 4. 실행
ollama launch claude --model gemma4:31b
```

### 중요 사항

- 외부 PC에는 **모델 다운로드가 필요 없음** (서버 PC의 GPU를 원격 사용)
- **Anthropic API 키 불필요** (로컬 Gemma 4 사용)
- 같은 네트워크(같은 Wi-Fi/공유기)면 **인증 없이** 바로 접속
- 외부 네트워크에서는 **API Key 필요** (게이트웨이에서 발급)

---

## 3. API Gateway

외부 PC가 서버 PC의 Gemma 4에 접근할 수 있도록 하는 프록시 서버.

### 기능

| 기능 | 설명 |
|------|------|
| API Key 인증 | Bearer 토큰 기반 인증 |
| 내부 IP 자동 허용 | 192.168.*, 172.*, 10.* 대역은 인증 없이 통과 |
| Rate Limiting | 키별 분당 요청 제한 (기본 30 req/min) |
| 엔드포인트 필터링 | 안전한 엔드포인트만 허용, 위험한 것 차단 |
| 스트리밍 지원 | Ollama 스트리밍 응답 프록시 |

### 허용/차단 엔드포인트

| 엔드포인트 | 상태 |
|-----------|------|
| /api/chat | 허용 |
| /api/generate | 허용 |
| /api/tags | 허용 |
| /api/show | 허용 |
| /api/ps | 허용 |
| /v1/chat/completions | 허용 |
| /api/delete | **차단** |
| /api/create | **차단** |
| /api/pull | **차단** |
| /api/push | **차단** |

### 키 관리

```powershell
# 키 생성
python gateway/server.py key create "이름" --limit 30

# 키 목록
python gateway/server.py key list

# 키 폐기
python gateway/server.py key revoke key_01
```

### 외부 네트워크에서 접속 (공유기 밖)

공유기 포트포워딩 설정이 필요합니다:
- 외부 포트 9000 → 내부 서버IP:9000

```bash
# 외부에서 접속 시 API Key 필요
export OLLAMA_HOST="http://공인IP:9000"
# Authorization 헤더는 ollama launch에서 자동 처리 안 됨
# curl로 직접 사용:
curl http://공인IP:9000/api/chat \
  -H "Authorization: Bearer gw-sk-xxxxx" \
  -d '{"model":"gemma4:31b","messages":[{"role":"user","content":"hello"}]}'
```

---

## 4. 파인튜닝 (사용하면서 모델 개선)

사용 중 쌓인 세션 데이터로 Gemma 4를 파인튜닝하여 점점 코딩 품질을 향상시킵니다.

### 요구사항 (서버 PC)

- Python 3.12+
- NVIDIA GPU (RTX 4090 권장)
- 디스크 130GB 여유 (모델 + 변환 임시파일)

### 초기 설정

```powershell
.\scripts\setup-finetune.ps1
```

### 파인튜닝 실행

```powershell
# 1. Claude Code 세션 로그에서 학습 데이터 추출
python finetune/export.py
# → ~/.gemma4-agent/finetune/data/train.jsonl 생성

# 통계만 확인
python finetune/export.py --stats

# 긍정 피드백 대화만 추출
python finetune/export.py --quality positive

# 2. QLoRA 파인튜닝 실행 (~30분)
python finetune/train.py
# → GGUF 파일 생성

# 하이퍼파라미터 조정
python finetune/train.py --epochs 2 --rank 64

# 3. 파인튜닝된 모델을 Ollama에 등록
python finetune/reload.py
# → sanity test 통과 시 자동 교체

# 4. 파인튜닝된 모델로 실행
$env:CLAUDE_OLLAMA_MODEL = "gemma4-finetuned:latest"
.\launch.cmd
```

### 파인튜닝 설정

| 항목 | 기본값 | 설명 |
|------|--------|------|
| 모델 | gemma4-31B-it (4bit) | Unsloth 사전양자화 |
| LoRA rank | 32 | 코딩 작업에 적합 |
| LoRA alpha | 64 | rank의 2배 |
| 학습률 | 2e-4 | 표준 시작점 |
| 배치 사이즈 | 1 | 24GB VRAM 제약 |
| 시퀀스 길이 | 4096 | 조정 가능 |
| 양자화 | Q4_K_M | Ollama용 GGUF |

---

## 프로젝트 구조

```
local-ai/
├── launch.cmd                  # Windows 실행기
├── launch.ps1                  # PowerShell 실행기
├── gateway/
│   ├── server.py               # API Gateway (인증, Rate Limit, 프록시)
│   └── requirements.txt        # Python 의존성 (aiohttp)
├── finetune/
│   ├── export.py               # Claude Code 세션 → 학습 데이터 변환
│   ├── train.py                # Unsloth QLoRA 파인튜닝
│   └── reload.py               # GGUF → Ollama 모델 교체
├── scripts/
│   └── setup-finetune.ps1      # Unsloth 환경 설정
└── README.md
```

## 데이터 저장 위치

```
~/.gemma4-agent/
├── gateway/
│   └── keys.json               # API 키
└── finetune/
    ├── data/
    │   └── train.jsonl          # 학습 데이터
    ├── runs/
    │   └── run_YYYYMMDD/        # 학습 기록
    └── models/
        └── gemma4-vXXX/         # 파인튜닝된 모델
```
