def pick_models(models_list):
    """
    Auto-select chat and embedding models from /v1/models response
    without caring about provider or brand names.
    """
    if not models_list:
        return None, None
        
    ids = [m.get("id", "") for m in models_list]
    
    # Heuristic: embeddings often contain 'embed' in the name
    embed_candidates = [m for m in ids if "embed" in m.lower()]
    embed_model = embed_candidates[0] if embed_candidates else None
    
    # Heuristic: chat models often have these indicators
    chat_keywords = [
        "chat", "instruct", "qwen", "llama", "phi", "mistral", 
        "gemma", "gpt", "claude", "command", "mixtral", "deepseek"
    ]
    
    chat_candidates = [
        m for m in ids 
        if any(keyword in m.lower() for keyword in chat_keywords)
        and "embed" not in m.lower()  # exclude embedding models
    ]
    
    # Prefer models with "instruct" or "chat" in name
    priority_chat = [m for m in chat_candidates if any(p in m.lower() for p in ["instruct", "chat"])]
    chat_model = (priority_chat or chat_candidates or ids)[0] if (priority_chat or chat_candidates or ids) else None
    
    return chat_model, embed_model

def validate_models(chat_model, embed_model):
    """
    Validate that we have usable models, provide helpful errors if not.
    """
    errors = []
    
    if not chat_model:
        errors.append("No chat model available. Set BRAINBOX_CHAT_MODEL or run configure_models.py")
        
    if not embed_model:
        errors.append("No embedding model available. Set BRAINBOX_EMBED_MODEL or run configure_models.py")
    
    if errors:
        error_msg = "\n".join(f"  • {err}" for err in errors)
        raise RuntimeError(f"Model configuration incomplete:\n{error_msg}")
    
    return True