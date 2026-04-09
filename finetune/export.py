"""
Claude Code 세션 로그 → 파인튜닝 학습 데이터 변환기

Claude Code가 저장하는 세션 파일 (.jsonl)을 읽어서
Unsloth/ShareGPT 형식의 학습 데이터로 변환합니다.
"""

import json
import os
import glob
import re
from pathlib import Path
from datetime import datetime


# Claude Code 세션 저장 위치
CLAUDE_SESSIONS_DIR = Path.home() / ".claude" / "projects"
OUTPUT_DIR = Path.home() / ".gemma4-agent" / "finetune" / "data"

# 긍정 피드백 패턴 (한국어 + 영어)
POSITIVE_PATTERNS = [
    r"좋아", r"완벽", r"고마워", r"감사", r"잘\s*했", r"맞아", r"정확",
    r"good", r"great", r"perfect", r"thanks", r"exactly", r"nice",
    r"👍", r"🎉", r"✅",
]

# 부정 피드백 패턴
NEGATIVE_PATTERNS = [
    r"아니", r"다시", r"틀렸", r"잘못", r"안\s*돼", r"에러",
    r"wrong", r"no\s*,", r"fix", r"retry", r"error", r"bug",
]


def find_session_files():
    """Claude Code 세션 파일 탐색"""
    sessions = []
    for pattern in [
        str(CLAUDE_SESSIONS_DIR / "**" / "*.jsonl"),
    ]:
        sessions.extend(glob.glob(pattern, recursive=True))

    # .claude/projects/*/sessions/ 하위 파일도 탐색
    alt_pattern = str(Path.home() / ".claude" / "**" / "sessions" / "**" / "*.jsonl")
    sessions.extend(glob.glob(alt_pattern, recursive=True))

    return list(set(sessions))


def parse_session(filepath):
    """세션 파일에서 대화 추출"""
    messages = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    messages.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"  [SKIP] {filepath}: {e}")
        return None
    return messages


def extract_conversations(messages):
    """세션 메시지에서 user/assistant 대화 쌍 추출"""
    conversations = []
    current_convo = []

    for msg in messages:
        # 다양한 세션 파일 포맷 처리
        role = msg.get("role", "")
        content = ""

        if isinstance(msg.get("content"), str):
            content = msg["content"]
        elif isinstance(msg.get("content"), list):
            # content가 배열인 경우 텍스트만 추출
            text_parts = []
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)
        elif msg.get("message"):
            # 다른 포맷
            content = msg["message"] if isinstance(msg["message"], str) else ""
            role = msg.get("type", role)

        if not content or not role:
            continue

        # role 정규화
        if role in ("human", "user"):
            role = "user"
        elif role in ("assistant", "model"):
            role = "assistant"
        else:
            continue

        current_convo.append({"role": role, "content": content.strip()})

    return current_convo


def analyze_feedback(conversation):
    """대화의 피드백 품질 분석"""
    if len(conversation) < 2:
        return "neutral"

    # 마지막 사용자 메시지 확인
    last_user_msgs = [m for m in conversation if m["role"] == "user"]
    if not last_user_msgs:
        return "neutral"

    last_msg = last_user_msgs[-1]["content"].lower()

    positive_score = sum(1 for p in POSITIVE_PATTERNS if re.search(p, last_msg, re.IGNORECASE))
    negative_score = sum(1 for p in NEGATIVE_PATTERNS if re.search(p, last_msg, re.IGNORECASE))

    if positive_score > negative_score:
        return "positive"
    elif negative_score > positive_score:
        return "negative"
    return "neutral"


def conversation_to_training(conversation):
    """대화를 학습 데이터 포맷으로 변환"""
    # user/assistant 쌍만 추출
    pairs = []
    i = 0
    while i < len(conversation) - 1:
        if conversation[i]["role"] == "user" and conversation[i + 1]["role"] == "assistant":
            pairs.append({
                "role": "user",
                "content": conversation[i]["content"]
            })
            pairs.append({
                "role": "assistant",
                "content": conversation[i + 1]["content"]
            })
        i += 1

    if not pairs:
        return None

    return {"conversations": pairs}


def export_training_data(min_quality="neutral", max_length=8000):
    """전체 세션을 학습 데이터로 변환"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "train.jsonl"

    session_files = find_session_files()
    print(f"Found {len(session_files)} session files")

    total = 0
    exported = 0
    skipped_quality = 0
    skipped_length = 0

    quality_levels = {"positive": 2, "neutral": 1, "negative": 0}
    min_level = quality_levels.get(min_quality, 1)

    with open(output_file, "w", encoding="utf-8") as out:
        for filepath in session_files:
            messages = parse_session(filepath)
            if not messages:
                continue

            conversation = extract_conversations(messages)
            if len(conversation) < 2:
                continue

            total += 1
            feedback = analyze_feedback(conversation)
            level = quality_levels.get(feedback, 1)

            if level < min_level:
                skipped_quality += 1
                continue

            training = conversation_to_training(conversation)
            if not training:
                continue

            # 길이 제한
            total_len = sum(len(m["content"]) for m in training["conversations"])
            if total_len > max_length:
                skipped_length += 1
                continue

            out.write(json.dumps(training, ensure_ascii=False) + "\n")
            exported += 1

    print(f"\nExport complete:")
    print(f"  Sessions found:    {total}")
    print(f"  Exported:          {exported}")
    print(f"  Skipped (quality): {skipped_quality}")
    print(f"  Skipped (length):  {skipped_length}")
    print(f"  Output:            {output_file}")

    return exported


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Claude Code sessions to training data")
    parser.add_argument("--quality", choices=["positive", "neutral", "negative"],
                        default="neutral", help="Minimum feedback quality (default: neutral)")
    parser.add_argument("--max-length", type=int, default=8000,
                        help="Max total character length per conversation (default: 8000)")
    parser.add_argument("--stats", action="store_true", help="Show stats only, don't export")
    args = parser.parse_args()

    if args.stats:
        files = find_session_files()
        print(f"Session files: {len(files)}")
        for f in files[:10]:
            msgs = parse_session(f)
            if msgs:
                convo = extract_conversations(msgs)
                feedback = analyze_feedback(convo)
                print(f"  {Path(f).name}: {len(convo)} msgs, feedback={feedback}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")
    else:
        export_training_data(args.quality, args.max_length)
