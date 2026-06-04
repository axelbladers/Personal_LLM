from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="NPC Dialogue LLM", description="Local API for NPC dialogue.")

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "npc-llm"
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"

class PlayerQuery(BaseModel):
    player_line: str


# Simple NPC Response Template System - Tavern Keeper
NPC_RESPONSES = {
    # Greetings
    "hello|hail|greet|welcome": [
        "Welcome to the Wandering Wyvern! What can I get you?",
        "Aye, welcome stranger. Sit, rest yourself. First drink's on the house.",
        "Hail there! Pull up a stool. You look like you've had a long road.",
    ],
    
    # Identity
    "who|name|are you": [
        "Name's Aldor. I run this tavern. Been here for nigh on twenty years.",
        "I'm the keeper of this fine establishment. Aldor, at your service.",
        "They call me Aldor. I pour the drinks and keep the peace 'round here.",
    ],
    
    # Location
    "where|location|town|city|place|millhaven": [
        "You're in Millhaven, an old town at the crossroads. Traders pass through here regular-like.",
        "This is Millhaven, friend. Been standing here since my great-grandfather's time.",
        "Welcome to Millhaven. Ancient town, old stories, good ale.",
    ],
    
    # Food and Drink - Buy/Order
    "buy|drink|food|eat|order|ale|beer|wine|menu|serve": [
        "We've got ale, mead, and wine. For food, there's bread, cheese, and stew. A copper for ale, silver for the good stuff.",
        "I serve the finest ale in three towns. Stew's hot if you're hungry. What'll it be?",
        "Ale's two coppers, mead's three. Bread and cheese are a copper. Stew's hearty and fills a belly.",
    ],
    
    # Price/Cost
    "price|cost|how much|copper|silver|gold|pay": [
        "Ale's cheap—two coppers. Mead costs three. A hot meal runs five.",
        "We're not greedy here. Ale is two coppers, mead is three, and the stew is five.",
        "Fair prices, friend. Everything you need for honest coin.",
    ],
    
    # Quest/Task/Job/Work - more specific
    "quest|task|work|job|hire|bounty|militia|work": [
        "Rumors travel through here. Some say there's bandits on the north road. Militia's looking for able fighters.",
        "There's always something. Merchants need guards, farms need hands. Ask around.",
        "Talk to the blacksmith about work, or the mayor if you're looking for something bigger. I just pour drinks.",
    ],
    
    # Rumors/News/Gossip
    "rumor|news|gossip|story|stories|tale|tales|heard|say|talk|legend": [
        "Well, bandits have been spotted north of here. The old mill's been quiet too—some say it's haunted.",
        "They say the forest's wilder these days. Something's stirring out there, if you ask me.",
        "I hear strange things from travelers. Lights in the woods, that sort of thing. Probably nothing.",
    ],
    
    # Room/Stay/Sleep/Rest
    "room|stay|sleep|bed|inn|rest|night|sleep": [
        "Rooms are upstairs. Two silver for a clean bed and breakfast. Stable's out back for your horse.",
        "Aye, we've got rooms. Good beds, clean sheets. Two silver for the night.",
        "There's beds upstairs. Clean and warm. Two silver, or three with a meal in the morning.",
    ],
    
    # Danger/Safety/Monster/Combat
    "danger|safe|monster|creature|fight|battle|enemy|threat|risky|perilous": [
        "The roads are mostly safe if you stick to the daylight. Don't go wandering the woods at night.",
        "There's been talk of something in the old ruins south of town. Wouldn't go looking for it, if I were you.",
        "Nothing too dangerous nearby, but keep your wits about you in the forest. Strange things happen there.",
    ],
    
    # Directions/Routes
    "direction|way|road|path|north|south|east|west|route|head": [
        "North leads to the trading post. East goes to the farms. South is the old ruins—don't go there.",
        "The main road heads north. If you want the forest, go east past the mill.",
        "Take the north road if you're heading to the city. East takes you toward the timber camps.",
    ],
    
    # Information/History/Background
    "history|old|ancient|story|legend|background|ruin|origin|past": [
        "Millhaven's old—been here since before my grandfather. There's history in these stones.",
        "This town's seen better days. Used to be a proper hub for trade. Still got good folk though.",
        "Ancient place, this. Some say there's old magic buried in the ruins. Folk don't talk about it much.",
    ],
    
    # Enemies/Bandits/Trouble/Militia
    "bandit|thief|guard|militia|trouble|problem|captain": [
        "Bandits? Aye, there's been some. Militia chases 'em off when they get bold. Roads aren't entirely safe.",
        "The law's thin on the ground out here. We look after ourselves mostly.",
        "There's a militia captain in town. He might have work for someone handy with a blade.",
    ],
    
    # Basic greetings
    "hi|hey|hallo": [
        "Well met, stranger. Take a seat.",
        "Aye, you alright?",
        "What brings you to my tavern?",
    ],
}


def get_template_response(player_text: str) -> str:
    """Return a response based on keyword matching."""
    import random
    player_lower = player_text.lower()
    
    # Check for matches in order of keyword group length (longer keywords first to prioritize specific matches)
    sorted_keywords = sorted(NPC_RESPONSES.keys(), key=lambda x: len(x.split("|")), reverse=True)
    
    for keyword_group in sorted_keywords:
        keywords = keyword_group.split("|")
        # Check for exact word matches or common phrases
        for keyword in keywords:
            if keyword in player_lower:
                responses = NPC_RESPONSES[keyword_group]
                return random.choice(responses)
    
    # Default responses if no keyword matches
    default_responses = [
        "You'll have to speak plainer than that, friend.",
        "Not sure I caught that. Say again?",
        "Eh? Didn't quite understand you there.",
        "Could you rephrase that? I'm just a humble tavern keeper.",
    ]
    return random.choice(default_responses)


def get_pad_token_id(tokenizer):
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return 0


def load_model():
    if MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists():
        print(f"Loading local model from {MODEL_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    else:
        print("Local model not found, loading pretrained distilgpt2.")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model


tokenizer, model = load_model()
pad_token_id = get_pad_token_id(tokenizer)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/chat")
def chat(query: PlayerQuery):
    player_text = query.player_line.strip()
    if not player_text:
        return JSONResponse(status_code=400, content={"detail": "player_line must not be empty"})

    # Use template-based response for reliability
    npc_response = get_template_response(player_text)

    return {"npc_response": npc_response}


@app.get("/")
def root():
    return FileResponse(INDEX_FILE)
