"""
Gemma 4 31B QLoRA 파인튜닝 스크립트

RTX 4090 24GB에서 Unsloth를 사용한 QLoRA 파인튜닝.
export.py로 생성한 train.jsonl을 학습합니다.
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# 경로 설정
DATA_DIR = Path.home() / ".gemma4-agent" / "finetune" / "data"
RUNS_DIR = Path.home() / ".gemma4-agent" / "finetune" / "runs"
MODELS_DIR = Path.home() / ".gemma4-agent" / "finetune" / "models"

# 기본 학습 설정
DEFAULT_CONFIG = {
    "model_name": "unsloth/gemma-4-31B-it-unsloth-bnb-4bit",
    "max_seq_length": 4096,
    "lora_rank": 32,
    "lora_alpha": 64,
    "learning_rate": 2e-4,
    "epochs": 1,
    "batch_size": 1,
    "gradient_accumulation": 8,
    "quantization": "q4_k_m",
}


def check_prerequisites():
    """사전 조건 확인"""
    # CUDA 확인
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available!")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu} ({vram:.0f}GB VRAM)")

    # 학습 데이터 확인
    train_file = DATA_DIR / "train.jsonl"
    if not train_file.exists():
        print(f"[ERROR] Training data not found: {train_file}")
        print("Run first: python finetune/export.py")
        sys.exit(1)

    sample_count = sum(1 for _ in open(train_file, encoding="utf-8"))
    print(f"Training data: {sample_count} samples")

    if sample_count < 10:
        print("[WARN] Very few samples. Consider collecting more data first.")

    return sample_count


def train(config=None):
    """QLoRA 파인튜닝 실행"""
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # 실행 디렉토리 생성
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 설정 저장
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Gemma 4 QLoRA Fine-tuning")
    print(f"  Run: {run_id}")
    print(f"  Config: {json.dumps(cfg, indent=2)}")
    print(f"{'='*50}\n")

    # Unsloth 임포트
    torch._dynamo.config.recompile_limit = 64

    from unsloth import FastVisionModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    # 1) 모델 로드
    print("[1/5] Loading model...")
    model, processor = FastVisionModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    # 2) LoRA 어댑터 설정
    print("[2/5] Setting up LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0,
        bias="none",
        random_state=3407,
        target_modules="all-linear",
    )

    # 3) 데이터 준비
    print("[3/5] Preparing training data...")
    processor = get_chat_template(processor, "gemma-4")
    tokenizer = processor.tokenizer

    dataset = load_dataset("json", data_files=str(DATA_DIR / "train.jsonl"), split="train")

    def format_data(examples):
        texts = []
        for convo in examples["conversations"]:
            text = tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            if text.startswith("<bos>"):
                text = text[5:]
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_data, batched=True)
    print(f"  Dataset size: {len(dataset)} samples")

    # 4) 학습
    print("[4/5] Training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation"],
            learning_rate=cfg["learning_rate"],
            optim="adamw_8bit",
            bf16=True,
            max_seq_length=cfg["max_seq_length"],
            num_train_epochs=cfg["epochs"],
            logging_steps=1,
            save_strategy="steps",
            save_steps=50,
            output_dir=str(run_dir / "checkpoints"),
            report_to="none",
        ),
    )

    # 모델 응답만 학습
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    print(f"  GPU memory reserved: {start_mem} GB / {gpu_stats.total_memory / 1024**3:.0f} GB")

    trainer.train()

    peak_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    print(f"  Peak memory: {peak_mem} GB")

    # LoRA 어댑터 저장
    model.save_pretrained(str(run_dir / "lora_adapter"))
    processor.save_pretrained(str(run_dir / "lora_adapter"))

    # 5) GGUF 변환
    print("[5/5] Exporting to GGUF...")
    gguf_dir = MODELS_DIR / f"gemma4-v{run_id}"
    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method=cfg["quantization"],
    )

    # GGUF 파일 찾기
    gguf_files = list(gguf_dir.glob("*.gguf"))
    if gguf_files:
        gguf_path = gguf_files[0]
        print(f"\n  GGUF saved: {gguf_path}")
        print(f"  Size: {gguf_path.stat().st_size / 1024**3:.1f} GB")

        # 경로 기록
        with open(MODELS_DIR / "latest.txt", "w") as f:
            f.write(str(gguf_path))

    print(f"\nTraining complete!")
    print(f"  Run directory: {run_dir}")
    print(f"  Next step: python finetune/reload.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 31B with QLoRA")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--rank", type=int, help="LoRA rank")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--seq-length", type=int, help="Max sequence length")
    args = parser.parse_args()

    config = {}
    if args.epochs:
        config["epochs"] = args.epochs
    if args.rank:
        config["lora_rank"] = args.rank
        config["lora_alpha"] = args.rank * 2
    if args.lr:
        config["learning_rate"] = args.lr
    if args.seq_length:
        config["max_seq_length"] = args.seq_length

    check_prerequisites()
    train(config)
