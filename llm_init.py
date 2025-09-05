"""
LLM Initialization - True Model-Agnostic Architecture
====================================================

Replaces all the coupled LLM imports with clean provider-neutral init.
No more hardcoded provider names in business logic.
"""

from config_loader import load_config
from llm_client import LLMClient
from model_select import pick_models, validate_models

def init_llm(argv=None):
    """
    Initialize LLM client with auto-detection and validation.
    Returns: (client, chat_model, embed_model) tuple
    """
    cfg = load_config(argv)
    client = LLMClient(cfg["base_url"], api_key=cfg["api_key"], timeout=cfg["timeout"])

    chat_model = cfg.get("chat_model")
    embed_model = cfg.get("embed_model")

    # Auto-detect models if not specified
    if not chat_model or not embed_model:
        print(f"[INFO] Auto-detecting models from {cfg['base_url']}")
        models_response = client.list_models()
        models_list = models_response.get("data", [])
        
        auto_chat, auto_embed = pick_models(models_list)
        chat_model = chat_model or auto_chat
        embed_model = embed_model or auto_embed
        
        if auto_chat and not cfg.get("chat_model"):
            print(f"[INFO] Auto-selected chat model: {auto_chat}")
        if auto_embed and not cfg.get("embed_model"):
            print(f"[INFO] Auto-selected embed model: {auto_embed}")

    # Validate we have usable models
    validate_models(chat_model, embed_model)
    
    return client, chat_model, embed_model

def create_legacy_config(client, chat_model, embed_model):
    """
    Create legacy config object for backward compatibility.
    Use this during transition period only.
    """
    class LegacyConfig:
        def __init__(self):
            self.base_url = client.base_url
            self.chat_model = chat_model
            self.embed_model = embed_model
            self.api_key = client.headers.get("Authorization", "").replace("Bearer ", "")
            self.timeout = client.timeout
    
    return LegacyConfig()