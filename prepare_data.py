"""
Prepare NPC dialogue data for training.

Reads data/sample_npc_dialogues.jsonl and writes a HuggingFace Dataset
to data/npc_dialogues_dataset.arrow that train_npc_llm.py can consume.

Usage:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# NPC system prompts — keep in sync with main.py and train_npc_llm.py
NPC_SYSTEMS = {
    "aldor":     "You are Aldor, a weathered tavern keeper in Millhaven. Gruff but fair. Short sentences, earthy idioms.",
    "seraphine": "You are Seraphine, an imperious court mage. Precise and slightly condescending. Elevated, formal register.",
    "brands":    "You are Brands, a battle-hardened mercenary captain. Blunt, loyal to coin. Short, clipped sentences.",
    "mira":      "You are Mira, a gentle wandering herbalist. Warm and curious. Soft, poetic speech.",
}


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning("Skipping line %d — JSON error: %s", i, e)
    return items


def format_example(item: dict) -> str | None:
    npc_id = item.get("npc_id", "aldor")
    system = NPC_SYSTEMS.get(npc_id, NPC_SYSTEMS["aldor"])
    conv   = item.get("conversation", [])

    lines = []
    for msg in conv:
        speaker = msg.get("speaker", "")
        text    = msg.get("text", "").strip()
        if speaker == "Player":
            lines.append(f"Player: {text}")
        elif speaker == "NPC":
            lines.append(f"NPC: {text}")

    if not lines:
        return None

    return f"[System: {system}]\n" + "\n".join(lines) + "\nNPC:"


def main():
    root       = Path(__file__).resolve().parents[1]
    data_file  = root / "data" / "sample_npc_dialogues.jsonl"
    output_dir = root / "data" / "npc_dialogues_dataset.arrow"

    if not data_file.exists():
        logging.error("Data file not found: %s", data_file)
        return

    raw = load_jsonl(data_file)
    logging.info("Read %d raw examples", len(raw))

    processed = []
    for item in raw:
        text = format_example(item)
        if text:
            processed.append({"text": text})

    if not processed:
        logging.error("No valid examples found.")
        return

    ds = Dataset.from_list(processed)
    ds.save_to_disk(str(output_dir))
    logging.info("Saved %d examples to %s", len(ds), output_dir)


if __name__ == "__main__":
    main()
