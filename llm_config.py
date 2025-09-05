# llm_config.py - compatibility shim for unified_brainbox.py
# DEPRECATED: Use config_loader.py + llm_client.py for new code

from config_loader import load_config

class BrainBoxLLMConfig:
    """Compatibility class for legacy imports"""
    def __init__(self, cfg=None):
        cfg = cfg or load_config()
        self.base_url   = cfg["base_url"]
        self.chat_model = cfg["chat_model"]
        self.embed_model= cfg["embed_model"]
        self.api_key    = cfg["api_key"]
        self.timeout    = cfg["timeout"]

# Legacy compatibility exports
CONFIG = BrainBoxLLMConfig()
BASE_URL = CONFIG.base_url
CHAT_MODEL = CONFIG.chat_model
EMBED_MODEL = CONFIG.embed_model
API_KEY = CONFIG.api_key
TIMEOUT = CONFIG.timeout