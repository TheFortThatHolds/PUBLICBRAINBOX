#!/usr/bin/env python3
"""
BrainBox Model Configuration Tool
Supports: LM Studio (local), OpenAI, Anthropic, any OpenAI-compatible API
"""

import asyncio, json, pathlib, httpx

PRESETS = {
    "lm_studio": {
        "base_url": "http://localhost:1234/v1",
        "auto_detect": True
    },
    "openai": {
        "base_url": "https://api.openai.com/v1", 
        "chat_model": "gpt-4o-mini",
        "coder_model": "gpt-4o",
        "embed_model": "text-embedding-3-small"
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "chat_model": "claude-3-haiku-20240307",
        "coder_model": "claude-3-5-sonnet-20241022", 
        "embed_model": "text-embedding-3-small"  # Fallback to OpenAI for embeddings
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "chat_model": "llama-3.1-8b-instant",
        "coder_model": "llama-3.1-70b-versatile",
        "embed_model": "text-embedding-3-small"
    }
}

async def auto_detect_models(base_url: str) -> dict:
    """Auto-detect available models from LM Studio API"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{base_url}/models")
            data = response.json()
            models = [m["id"] for m in data["data"]]
            
            # Smart model selection
            chat_model = None
            coder_model = None
            embed_model = None
            
            for model in models:
                model_lower = model.lower()
                # Embedding models
                if "embed" in model_lower:
                    embed_model = model
                # Coding models (larger, specialized)
                elif any(x in model_lower for x in ["coder", "code", "30b", "34b", "70b"]):
                    if not coder_model:  # Take first large/coder model
                        coder_model = model
                # Chat models (smaller, general purpose)
                elif any(x in model_lower for x in ["4b", "7b", "instruct", "chat"]):
                    if not chat_model:  # Take first smaller model
                        chat_model = model
            
            # Fallbacks
            if not chat_model and models:
                chat_model = models[0]  # Use first available
            if not coder_model:
                coder_model = chat_model  # Fallback to chat model
            if not embed_model:
                embed_model = "text-embedding-ada-002"  # OpenAI fallback
                
            return {
                "chat_model": chat_model,
                "coder_model": coder_model, 
                "embed_model": embed_model,
                "available_models": models
            }
    except Exception as e:
        print(f"Model auto-detection failed: {e}")
        return {"error": str(e)}

async def main():
    print("🧠 BrainBox Model Configuration")
    print("=" * 40)
    
    print("Choose configuration:")
    print("1. Auto-detect from LM Studio (recommended)")
    print("2. OpenAI API")
    print("3. Anthropic API") 
    print("4. Groq API")
    print("5. Custom API endpoint")
    
    choice = input("\nSelect (1-5): ").strip()
    
    # Create basic config structure
    config = {
        "api": {
            "base_url": "http://localhost:1234/v1",
            "chat_model": "gpt-3.5-turbo",
            "embed_model": "text-embedding-ada-002",
            "timeout": 30
        }
    }
    
    if choice == "1":
        # Auto-detect LM Studio
        url = input(f"LM Studio URL [{PRESETS['lm_studio']['base_url']}]: ").strip()
        if not url:
            url = PRESETS['lm_studio']['base_url']
            
        print("🔍 Detecting models...")
        result = await auto_detect_models(url)
        
        if "error" in result:
            print(f"❌ Detection failed: {result['error']}")
            return
            
        print(f"✅ Found {len(result['available_models'])} models:")
        for model in result['available_models']:
            print(f"  • {model}")
        print()
        print(f"🎯 Auto-selected:")
        print(f"  💬 Chat: {result['chat_model']}")
        print(f"  🔍 Embed: {result['embed_model']}")
        
        if input("\nUse these? (y/n): ").lower().startswith('y'):
            config["api"].update({
                "base_url": url,
                "chat_model": result['chat_model'],
                "embed_model": result['embed_model']
            })
            
    elif choice in ["2", "3", "4"]:
        # Preset APIs
        preset_names = {"2": "openai", "3": "anthropic", "4": "groq"}
        preset = PRESETS[preset_names[choice]]
        
        print(f"🌐 Configuring {preset_names[choice].upper()} API")
        api_key = input(f"API Key: ").strip()
        
        config["api"].update(preset)
        config["api"]["api_key"] = api_key
        
    elif choice == "5":
        # Custom endpoint
        print("🔧 Custom API Configuration")
        url = input("API Base URL: ").strip()
        chat_model = input("Chat Model ID: ").strip()
        embed_model = input("Embedding Model ID: ").strip()
        api_key = input("API Key (optional): ").strip()
        
        config["api"].update({
            "base_url": url,
            "chat_model": chat_model,
            "embed_model": embed_model
        })
        if api_key:
            config["api"]["api_key"] = api_key
    else:
        print("❌ Invalid choice")
        return
        
    # Save config
    config_path = pathlib.Path("brainbox_config.json")
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration saved!")
    print(f"\n🎉 Ready to use! Your BrainBox is configured with:")
    print(f"  📍 API: {config['api']['base_url']}")
    print(f"  💬 Chat: {config['api']['chat_model']}")
    print(f"  🔍 Embed: {config['api']['embed_model']}")

if __name__ == "__main__":
    asyncio.run(main())