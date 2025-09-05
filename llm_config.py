# llm_config.py — compatibility shim
import json, pathlib, os

CFG_PATH = pathlib.Path("brainbox_config.json")

def _load():
    if CFG_PATH.exists():
        with CFG_PATH.open() as f:
            return json.load(f)
    return {
        "api": {
            "base_url": "http://localhost:1234/v1",
            "chat_model": "qwen3-4b-instruct-2507",
            "embed_model": "text-embedding-nomic-embed-text-v1.5",
            "api_key": os.getenv("OPENAI_API_KEY", "")
        }
    }

CFG = _load()

# Expose values for old code
BASE_URL   = CFG["api"].get("base_url")
CHAT_MODEL = CFG["api"].get("chat_model")
EMBED_MODEL= CFG["api"].get("embed_model")
API_KEY    = CFG["api"].get("api_key", "")
TIMEOUT    = CFG["api"].get("timeout", 30)
