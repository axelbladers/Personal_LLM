import logging
from pathlib import Path

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)


def main():
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models" / "npc-llm"
    dataset_dir = project_root / "data" / "npc_dialogues_dataset.arrow"

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    dataset = load_from_disk(str(dataset_dir))

    def tokenize_examples(examples):
        outputs = tokenizer(examples["text"], truncation=True, max_length=256)
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized = dataset.map(tokenize_examples, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(model_dir),
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        learning_rate=5e-5,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Започва обучение на NPC LLM...")
    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"Готов модел записан в: {model_dir}")


if __name__ == "__main__":
    main()
