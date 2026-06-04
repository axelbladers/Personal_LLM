"""
NPC Dialogue Server — powered by Ollama (local LLM inference).

Requires Ollama running locally:
    ollama serve
    ollama pull llama3          # or mistral, phi3, gemma2, etc.

Start this server:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ─── Config ──────────────────────────────────────────────────────────────────

OLLAMA_BASE   = "http://localhost:11434"
OLLAMA_MODEL  = "llama3"          # change to "mistral", "phi3", "gemma2", etc.
MAX_HISTORY   = 12                # message pairs kept per session
STATIC_DIR    = Path(__file__).resolve().parent / "static"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── NPC Definitions ─────────────────────────────────────────────────────────

NPC_ROSTER: dict[str, dict] = {
    "aldor": {
        "name": "Aldor",
        "title": "Tavern Keeper",
        "location": "The Wandering Wyvern, Millhaven",
        "avatar": "🍺",
        "system": (
            "You are Aldor, a weathered tavern keeper in the medieval fantasy town of Millhaven. "
            "You have run The Wandering Wyvern tavern for over twenty years. "
            "You are gruff but fair, fond of ale and a good story. "
            "You speak in a casual, slightly archaic tone — short sentences, earthy idioms. "
            "You know local gossip, road conditions, prices of food and drink, and rumours of nearby dangers. "
            "You do NOT know about modern technology, events outside your region, or anything beyond your world. "
            "Stay firmly in character at all times. Never break character or acknowledge being an AI. "
            "Keep responses to 2–4 sentences unless the player asks for a long story."
        ),
    },
    "seraphine": {
        "name": "Seraphine",
        "title": "Court Mage",
        "location": "The Royal Academy of Aethermoor",
        "avatar": "🔮",
        "system": (
            "You are Seraphine, an imperious and brilliant court mage at the Royal Academy of Aethermoor. "
            "You are precise, a little condescending, and obsessed with arcane lore. "
            "You speak in an elevated, slightly formal register — you consider yourself intellectually superior "
            "to most people you meet, though you have a dry sense of humour. "
            "You know about spells, magical artifacts, the Academy's politics, and ancient history. "
            "You find small talk tedious and prefer discussing matters of magical importance. "
            "Stay firmly in character. Never break character or acknowledge being an AI. "
            "Keep responses to 2–4 sentences unless explaining a spell or lore in detail."
        ),
    },
    "brands": {
        "name": "Brands",
        "title": "Mercenary Captain",
        "location": "Ironwatch Garrison, Northern Reach",
        "avatar": "⚔️",
        "system": (
            "You are Brands, a battle-hardened mercenary captain stationed at Ironwatch Garrison on the northern frontier. "
            "You are blunt, loyal to coin, and have seen too much war to be easily impressed. "
            "You speak in short, clipped sentences. You respect competence, distrust nobles, and have a gallows sense of humour. "
            "You know about weapons, combat tactics, troop movements, bandit activity, and frontier dangers. "
            "You have little patience for nonsense and will say so. "
            "Stay firmly in character. Never break character or acknowledge being an AI. "
            "Keep responses to 1–3 sentences unless describing a battle or giving tactical advice."
        ),
    },
    "mira": {
        "name": "Mira",
        "title": "Wandering Herbalist",
        "location": "The Greenwood Road",
        "avatar": "🌿",
        "system": (
            "You are Mira, a gentle and knowledgeable herbalist who travels the roads selling potions and remedies. "
            "You are warm, curious, and deeply connected to nature and folk medicine. "
            "You speak softly, with a poetic quality — you notice small details and make gentle observations. "
            "You know about healing herbs, poisons, forest paths, local folklore, and the healing arts. "
            "You distrust violence but will help those who are hurt. "
            "Stay firmly in character. Never break character or acknowledge being an AI. "
            "Keep responses to 2–4 sentences."
        ),
    },
}

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="NPC Dialogue", description="Local LLM-powered NPC chat via Ollama.")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory conversation store: {session_id: {npc_id: deque[{role, content}]}}
_sessions: dict[str, dict[str, deque]] = {}


# ─── Schemas ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    npc_id:     str          = Field(..., description="NPC identifier, e.g. 'aldor'")
    message:    str          = Field(..., description="Player's message")
    session_id: str          = Field(..., description="Session identifier for history")
    stream:     bool         = Field(True,  description="Stream the response token-by-token")


class NPCInfo(BaseModel):
    id:       str
    name:     str
    title:    str
    location: str
    avatar:   str


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_history(session_id: str, npc_id: str) -> deque:
    return _sessions.setdefault(session_id, {}).setdefault(npc_id, deque(maxlen=MAX_HISTORY * 2))


async def _check_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{OLLAMA_BASE}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def _stream_ollama(system: str, messages: list[dict]) -> AsyncIterator[str]:
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream":   True,
        "options": {
            "temperature":     0.75,
            "top_p":           0.9,
            "repeat_penalty":  1.1,
            "num_predict":     300,
        },
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", f"{OLLAMA_BASE}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            import json
            async for raw_line in resp.aiter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue


async def _call_ollama(system: str, messages: list[dict]) -> str:
    """Non-streaming call — collects full response."""
    result = []
    async for token in _stream_ollama(system, messages):
        result.append(token)
    return "".join(result)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/npcs", response_model=list[NPCInfo])
def list_npcs():
    return [
        NPCInfo(id=k, name=v["name"], title=v["title"], location=v["location"], avatar=v["avatar"])
        for k, v in NPC_ROSTER.items()
    ]


@app.get("/health")
async def health():
    ok = await _check_ollama()
    model_info = {"model": OLLAMA_MODEL, "ollama_reachable": ok}
    if not ok:
        model_info["hint"] = "Run: ollama serve && ollama pull " + OLLAMA_MODEL
    return model_info


@app.post("/chat")
async def chat(req: ChatRequest):
    npc_id = req.npc_id.lower().strip()
    if npc_id not in NPC_ROSTER:
        raise HTTPException(404, f"NPC '{npc_id}' not found. Available: {list(NPC_ROSTER)}")

    message = req.message.strip()
    if not message:
        raise HTTPException(400, "message must not be empty")

    npc   = NPC_ROSTER[npc_id]
    hist  = _get_history(req.session_id, npc_id)

    # Append player turn
    hist.append({"role": "user", "content": message})
    messages = list(hist)

    if req.stream:
        # We need to collect the reply to save it to history too
        reply_parts: list[str] = []

        async def event_stream():
            async for token in _stream_ollama(npc["system"], messages):
                reply_parts.append(token)
                yield token
            # After stream ends, persist assistant turn
            full_reply = "".join(reply_parts)
            hist.append({"role": "assistant", "content": full_reply})

        return StreamingResponse(event_stream(), media_type="text/plain")

    else:
        try:
            reply = await _call_ollama(npc["system"], messages)
        except httpx.ConnectError:
            raise HTTPException(503, "Cannot reach Ollama. Is it running? Try: ollama serve")
        except Exception as e:
            log.exception("Ollama error")
            raise HTTPException(500, str(e))

        hist.append({"role": "assistant", "content": reply})
        return {
            "npc_name":  npc["name"],
            "npc_id":    npc_id,
            "response":  reply,
            "session_id": req.session_id,
        }


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    cleared = list(_sessions.pop(session_id, {}).keys())
    return {"session_id": session_id, "cleared_npcs": cleared}
