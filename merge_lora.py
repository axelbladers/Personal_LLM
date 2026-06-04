"""
Merge LoRA adapters into the base model after fine-tuning.

Run this after train_npc_llm.py completes:
    python scripts/merge_lora.py

Output: models/npc-llm-merged  (full HuggingFace model, ready for GGUF conversion)

Note: Merging requires the full model in fp16 on CPU.
      Expect ~32 GB RAM usage for a 7–8B model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

BASE_MODEL  = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_DIR    = Path(__file__).resolve().parents[1] / "models" / "npc-llm-lora"
MERGED_DIR  = Path(__file__).resolve().parents[1] / "models" / "npc-llm-merged"


def main():
    logging.info("Loading base model on CPU (this uses ~16 GB RAM)…")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    logging.info("Applying LoRA adapters from %s…", LORA_DIR)
    model = PeftModel.from_pretrained(base, str(LORA_DIR))

    logging.info("Merging and unloading (this may take a few minutes)…")
    merged = model.merge_and_unload()

    logging.info("Saving merged model to %s…", MERGED_DIR)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(str(LORA_DIR)).save_pretrained(str(MERGED_DIR))

    logging.info(
        "\nMerge complete! Next:\n"
        "  cd llama.cpp\n"
        "  python convert_hf_to_gguf.py ../models/npc-llm-merged \\\n"
        "      --outfile ../models/npc-llm-f16.gguf --outtype f16\n"
        "  ./build/bin/llama-quantize ../models/npc-llm-f16.gguf \\\n"
        "      ../models/npc-llm-Q4_K_M.gguf Q4_K_M\n"
        "  ollama create npc-model -f Modelfile"
    )


if __name__ == "__main__":
    main()
