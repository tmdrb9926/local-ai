"""
파인튜닝된 GGUF 모델을 Ollama에 등록하고 교체

1. GGUF 파일 검증
2. Modelfile 생성
3. ollama create로 등록
4. 간단한 sanity test
5. 이전 모델 백업
"""

import os
import sys
import json
import struct
import subprocess
import requests
from pathlib import Path
from datetime import datetime

MODELS_DIR = Path.home() / ".gemma4-agent" / "finetune" / "models"
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "gemma4-finetuned"

# Sanity test 프롬프트
SANITY_PROMPTS = [
    "What is 2 + 2?",
    "Write a Python hello world.",
    "Explain what a variable is in one sentence.",
]


def validate_gguf(path):
    """GGUF 파일 기본 검증"""
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return False

    size_gb = path.stat().st_size / 1024**3
    if size_gb < 1:
        print(f"[ERROR] File too small ({size_gb:.2f} GB)")
        return False

    # GGUF 매직 넘버 확인
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"[ERROR] Not a valid GGUF file (magic: {magic})")
            return False

    print(f"  Valid GGUF: {path.name} ({size_gb:.1f} GB)")
    return True


def find_latest_gguf():
    """최신 GGUF 파일 찾기"""
    latest_file = MODELS_DIR / "latest.txt"
    if latest_file.exists():
        path = Path(latest_file.read_text().strip())
        if path.exists():
            return path

    # 모든 GGUF 파일 중 가장 최근 것
    gguf_files = list(MODELS_DIR.rglob("*.gguf"))
    if not gguf_files:
        return None
    return max(gguf_files, key=lambda p: p.stat().st_mtime)


def create_modelfile(gguf_path, output_path):
    """Ollama Modelfile 생성"""
    content = f"""FROM {gguf_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 32768
PARAMETER stop "<|turn>user"
"""
    output_path.write_text(content)
    print(f"  Modelfile: {output_path}")


def ollama_create(model_name, modelfile_path):
    """ollama create 실행"""
    print(f"  Creating model '{model_name}'...")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] ollama create failed: {result.stderr}")
        return False
    print(f"  Model created: {model_name}")
    return True


def sanity_test(model_name):
    """기본 품질 테스트"""
    print(f"\n  Running sanity tests on '{model_name}'...")
    passed = 0

    for prompt in SANITY_PROMPTS:
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"num_predict": 100},
                },
                timeout=120,
            )
            data = resp.json()
            content = data.get("message", {}).get("content", "")

            if len(content.strip()) > 5:
                print(f"    PASS: '{prompt[:30]}...' → '{content[:50]}...'")
                passed += 1
            else:
                print(f"    FAIL: '{prompt[:30]}...' → empty/short response")

        except Exception as e:
            print(f"    FAIL: '{prompt[:30]}...' → {e}")

    print(f"  Sanity: {passed}/{len(SANITY_PROMPTS)} passed")
    return passed >= 2  # 3개 중 2개 이상 통과


def reload_model():
    """전체 모델 교체 프로세스"""
    print("=" * 50)
    print("  Gemma 4 Model Reload")
    print("=" * 50)

    # 1. 최신 GGUF 찾기
    print("\n[1/4] Finding latest GGUF...")
    gguf_path = find_latest_gguf()
    if not gguf_path:
        print("[ERROR] No GGUF files found. Run train.py first.")
        sys.exit(1)

    # 2. 검증
    print("\n[2/4] Validating...")
    if not validate_gguf(gguf_path):
        sys.exit(1)

    # 3. Ollama에 등록
    print("\n[3/4] Registering with Ollama...")
    modelfile_path = gguf_path.parent / "Modelfile"
    create_modelfile(gguf_path, modelfile_path)

    version = datetime.now().strftime("%Y%m%d")
    versioned_name = f"{MODEL_NAME}:v{version}"

    if not ollama_create(versioned_name, modelfile_path):
        sys.exit(1)

    # latest 태그도 생성
    ollama_create(f"{MODEL_NAME}:latest", modelfile_path)

    # 4. Sanity test
    print("\n[4/4] Testing...")
    if sanity_test(versioned_name):
        print(f"\n  Model ready: {versioned_name}")
        print(f"\n  Use with Claude Code:")
        print(f"    $env:CLAUDE_OLLAMA_MODEL = '{versioned_name}'")
        print(f"    .\\launch.cmd")
    else:
        print(f"\n  [WARN] Sanity test failed. Consider rolling back.")
        print(f"    ollama rm {versioned_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reload fine-tuned model into Ollama")
    parser.add_argument("--gguf", type=str, help="Path to specific GGUF file")
    parser.add_argument("--name", type=str, default=MODEL_NAME, help="Model name in Ollama")
    parser.add_argument("--skip-test", action="store_true", help="Skip sanity test")
    args = parser.parse_args()

    if args.name:
        MODEL_NAME = args.name

    reload_model()
