# Personal_LLM — NPC Dialogue System

A local NPC dialogue system powered by a real instruction-tuned LLM via **Ollama**. No cloud, no API keys, no keyword matching — just a genuine language model running on your machine.

---

## How It Works

```
Player input → FastAPI (app/main.py)
                 └→ Ollama (local LLM) → streamed response → browser UI
```

Each NPC has a **system prompt** that defines their personality, speech style, and knowledge. The LLM stays in character across a full conversation using per-session history. Responses stream token-by-token to the UI.

---

## Quick Start (Recommended Path)

### 1. Install Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: download from https://ollama.ai
```

### 2. Pull a model

```bash
ollama pull llama3          # ~4.7 GB — best quality
# or
ollama pull mistral         # ~4.1 GB — slightly faster
# or
ollama pull phi3            # ~2.2 GB — for low VRAM
```

### 3. Start Ollama

```bash
ollama serve
```

### 4. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Start the app

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open the UI

```
http://127.0.0.1:8000
```

Select an NPC and start talking. Responses stream in real time.

---

## Changing the Model

Edit `OLLAMA_MODEL` at the top of `app/main.py`:

```python
OLLAMA_MODEL = "llama3"   # or "mistral", "phi3", "gemma2", etc.
```

Any model you've pulled with `ollama pull <name>` will work.

---

## NPC Characters

Four characters are included out of the box:

| NPC | Role | Personality |
|-----|------|-------------|
| **Aldor** | Tavern Keeper | Gruff, practical, knows local gossip |
| **Seraphine** | Court Mage | Imperious, precise, obsessed with arcane lore |
| **Brands** | Mercenary Captain | Blunt, battle-hardened, respects competence |
| **Mira** | Herbalist | Warm, nature-connected, poetic speech |

To add a new NPC, edit the `NPC_ROSTER` dict in `app/main.py`.

---

## API

### `POST /chat`

```json
{
  "npc_id":     "aldor",
  "message":    "Do you know anything about the bandits?",
  "session_id": "player_001",
  "stream":     true
}
```

With `"stream": true`, returns a plain-text streaming response.  
With `"stream": false`, returns:

```json
{
  "npc_name":   "Aldor",
  "npc_id":     "aldor",
  "response":   "Aye, nasty business...",
  "session_id": "player_001"
}
```

### `GET /npcs` — list all NPCs
### `GET /health` — check Ollama connection
### `DELETE /session/{session_id}` — clear conversation history

---

## Fine-Tuning (Optional)

If you want to train a custom model on your own dialogue data:

### 1. Add training data

Edit `data/sample_npc_dialogues.jsonl`. Each line:

```json
{"npc_id": "aldor", "conversation": [
  {"speaker": "Player", "text": "..."},
  {"speaker": "NPC",    "text": "..."}
]}
```

### 2. Prepare data

```bash
python scripts/prepare_data.py
```

### 3. Fine-tune with QLoRA

```bash
# Install training deps first:
pip install torch transformers datasets accelerate peft bitsandbytes trl

# Requires HuggingFace login + Meta license for Llama-3:
huggingface-cli login
python scripts/train_npc_llm.py
```

Runs on 8–16 GB VRAM. Takes 1–3 hours for a few hundred examples.

### 4. Merge adapters

```bash
python scripts/merge_lora.py
```

### 5. Convert to GGUF and load into Ollama

```bash
# Requires llama.cpp (https://github.com/ggerganov/llama.cpp)
cd llama.cpp
python convert_hf_to_gguf.py ../models/npc-llm-merged \
    --outfile ../models/npc-llm-f16.gguf --outtype f16
./build/bin/llama-quantize ../models/npc-llm-f16.gguf \
    ../models/npc-llm-Q4_K_M.gguf Q4_K_M

# Register with Ollama
ollama create my-npc-model -f Modelfile
```

Then set `OLLAMA_MODEL = "my-npc-model"` in `app/main.py`.

---

## Project Structure

```
app/
  main.py              ← FastAPI server + NPC roster + Ollama integration
  static/index.html    ← Chat UI (streaming, NPC selector, dark fantasy theme)
data/
  sample_npc_dialogues.jsonl   ← Training examples
scripts/
  prepare_data.py      ← Convert JSONL to HuggingFace Dataset
  train_npc_llm.py     ← QLoRA fine-tuning
  merge_lora.py        ← Merge LoRA adapters into base model
requirements.txt       ← Runtime deps (training deps commented out)
```
