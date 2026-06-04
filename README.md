# Personal_LLM

## Project Overview
This repository demonstrates how to build, train, and deploy a local NPC dialogue model. The goal is to create a self-hosted chat system for game NPCs that runs locally without relying on external cloud services.

## What is included
- `data/sample_npc_dialogues.jsonl` - sample NPC dialogue data in English.
- `scripts/prepare_data.py` - prepares the dataset for training.
- `scripts/train_npc_llm.py` - trains a local language model using Hugging Face Transformers.
- `app/main.py` - FastAPI server for the NPC chat application.
- `app/static/index.html` - simple web UI for interacting with the NPC.
- `requirements.txt` - Python dependencies.
- `.gitignore` - excludes temporary files and model artifacts.

## Notes
This is a simple starter demo. The app uses `distilgpt2` by default and a basic NPC prompt, so it is best for testing simple conversational behavior. For better results, train the local model with more Skyrim-style dialogue examples or replace `distilgpt2` with a larger model.

---

## Setup
1. Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preparation
1. Add or update sample dialogue lines in `data/sample_npc_dialogues.jsonl`.
2. Run:

```bash
python scripts/prepare_data.py
```

This script converts the JSONL data into a dataset format that the training script can use.

---

## Training
1. Run:

```bash
python scripts/train_npc_llm.py
```

2. The trained model will be saved to `models/npc-llm`.
3. More training data and more epochs will improve the NPC responses.

> Note: If you have a GPU available, training will run faster, but this example can also run on CPU.

---

## How It Works

The NPC dialogue system uses a simple **keyword-matching template system**:

1. **Keywords**: `hello`, `who`, `danger`, `work`, `gold`
2. **Responses**: Each keyword has multiple pre-written responses that are randomly selected
3. **Fallback**: If no keyword matches, a default response is used

This approach is reliable and produces consistent Skyrim-style dialogue without the complexity of fine-tuning a large language model.

---

## Example Conversations

```
Player: Who are you?
NPC: I'm a guard of this village, watching over travelers and the road.

Player: Is there danger nearby?
NPC: The forest is quiet now, but keep your sword ready at dusk.

Player: Can you help me find work?
NPC: Visit the blacksmith, he could use help hauling ore.
```

## Local Deployment
1. Start the app:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. Open the GUI in your browser at:

```text
http://127.0.0.1:8000
```

3. Type a player line and click **Send**. The NPC will respond based on keywords in your input.

4. To use the API directly:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"player_line": "Who are you?"}'
```

---
- Add more NPC dialogue examples.
- Create different NPC personalities for merchants, guards, and wizards.
- Use a larger pretrained model or fine-tune a local model for better quality.
- Add persistent chat history and context management.

---

## Summary
- `prepare_data.py` converts JSONL to a dataset for training.
- `train_npc_llm.py` trains a language model with Transformers.
- `app/main.py` provides a local REST API and web interface.
- `app/static/index.html` offers a simple English UI for NPC chat.

Ready to use the local NPC chat demo in English.
