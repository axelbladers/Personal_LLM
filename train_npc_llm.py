"""
NPC LLM Fine-Tuning Script — QLoRA + SFT
=========================================
Uses QLoRA (4-bit quantization + LoRA adapters) to fine-tune a capable
instruction model on NPC dialogue data. Works on 8–16 GB VRAM.

Requirements (install separately from main requirements.txt):
    pip install torch transformers datasets accelerate peft bitsandbytes trl

Usage:
    python scripts/train_npc_llm.py

The fine-tuned model is saved to models/npc-llm-lora (LoRA adapters only).
See the README for instructions on merging and converting to GGUF.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_MODEL  = "meta-llama/Meta-Llama-3-8B-Instruct"
# Alternatives (no license required):
#   "mistralai/Mistral-7B-Instruct-v0.3"
#   "microsoft/Phi-3-mini-4k-instruct"

DATA_FILE   = Path(__file__).resolve().parents[1] / "data" / "sample_npc_dialogues.jsonl"
OUTPUT_DIR  = Path(__file__).resolve().parents[1] / "models" / "npc-llm-lora"

LORA_RANK       = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
EPOCHS          = 3
BATCH_SIZE      = 2
GRAD_ACCUM      = 4       # effective batch = 8
LEARNING_RATE   = 2e-4
MAX_SEQ_LENGTH  = 1024
TEST_SPLIT      = 0.1

# ─── NPC system prompts (same as in main.py) ──────────────────────────────────

NPC_SYSTEMS = {
    "aldor":     "You are Aldor, a weathered tavern keeper in the medieval fantasy town of Millhaven. You are gruff but fair. You speak in a casual, slightly archaic tone — short sentences, earthy idioms.",
    "seraphine": "You are Seraphine, an imperious court mage at the Royal Academy. You are precise and a little condescending. You speak in an elevated, formal register.",
    "brands":    "You are Brands, a battle-hardened mercenary captain. You are blunt and loyal to coin. You speak in short, clipped sentences.",
    "mira":      "You are Mira, a gentle wandering herbalist. You are warm and curious. You speak softly, with a poetic quality.",
}


def format_llama3(system: str, user: str, assistant: str) -> str:
    """Format a single exchange in Llama-3 ChatML format."""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{assistant}<|eot_id|>"
    )


def format_mistral(system: str, user: str, assistant: str) -> str:
    """Format a single exchange in Mistral/Alpaca format."""
    return f"[INST] {system}\n\n{user} [/INST] {assistant}</s>"


def load_dataset() -> Dataset:
    examples = []
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            npc_id = item.get("npc_id", "aldor")
            system = NPC_SYSTEMS.get(npc_id, NPC_SYSTEMS["aldor"])
            conv = item["conversation"]
            for i in range(len(conv) - 1):
                if conv[i]["speaker"] == "Player" and conv[i + 1]["speaker"] == "NPC":
                    user_text = conv[i]["text"]
                    npc_text  = conv[i + 1]["text"]
                    # Use Llama-3 format; swap for format_mistral if using Mistral
                    text = format_llama3(system, user_text, npc_text)
                    examples.append({"text": text})

    logging.info("Loaded %d training examples", len(examples))
    return Dataset.from_list(examples)


def main():
    dataset = load_dataset()
    split   = dataset.train_test_split(test_size=TEST_SPLIT, seed=42)

    logging.info("Loading base model: %s", BASE_MODEL)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        # Target all linear layers for maximum expressiveness
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    logging.info("Starting training…")
    trainer.train()

    logging.info("Saving LoRA adapters to %s", OUTPUT_DIR)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    logging.info(
        "\nDone! Next steps:\n"
        "  1. Merge adapters: python scripts/merge_lora.py\n"
        "  2. Convert to GGUF: see README for llama.cpp instructions\n"
        "  3. Register with Ollama: ollama create npc-model -f Modelfile"
    )


if __name__ == "__main__":
    main()
