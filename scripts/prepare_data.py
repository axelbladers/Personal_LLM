import json
from pathlib import Path

from datasets import Dataset


def load_jsonl(path: Path):
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def format_conversation(conversation):
    # Бележка: форматираме диалога като редове, за да бъде удобен за обучение.
    lines = []
    for message in conversation:
        lines.append(f"{message['speaker']}: {message['text']}")
    lines.append("NPC:")
    return "\n".join(lines)


def main():
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "sample_npc_dialogues.jsonl"
    output_path = Path(__file__).resolve().parents[1] / "data" / "npc_dialogues_dataset.arrow"

    examples = load_jsonl(dataset_path)
    processed = [{"text": format_conversation(item["conversation"])} for item in examples]

    ds = Dataset.from_list(processed)
    ds.save_to_disk(str(output_path))
    print(f"Записах подготвения набор: {output_path}")


if __name__ == "__main__":
    main()
