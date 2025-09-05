import os
import json
import pathlib
import argparse

def _env(name, default=None):
    return os.getenv(name, default)

def _load_json_candidates():
    for p in [
        pathlib.Path("brainbox_config.json"),
        pathlib.Path(__file__).parent / "brainbox_config.json",
    ]:
        if p.exists():
            with p.open() as f:
                return json.load(f)
    return {}

def load_config(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--chat-model", dest="chat_model")
    parser.add_argument("--embed-model", dest="embed_model")
    parser.add_argument("--api-key", dest="api_key")
    args, _ = parser.parse_known_args(argv)

    cfg_json = _load_json_candidates()
    api = cfg_json.get("api", {})

    base_url   = args.base_url   or _env("BRAINBOX_BASE_URL",   api.get("base_url", "http://localhost:1234/v1"))
    chat_model = args.chat_model or _env("BRAINBOX_CHAT_MODEL", api.get("chat_model"))
    embed_model= args.embed_model or _env("BRAINBOX_EMBED_MODEL", api.get("embed_model"))
    api_key    = args.api_key    or _env("BRAINBOX_API_KEY",    api.get("api_key", "")) or _env("OPENAI_API_KEY", "")

    return {
        "base_url": base_url.rstrip("/") if base_url else "http://localhost:1234/v1",
        "chat_model": chat_model,
        "embed_model": embed_model,
        "api_key": api_key,
        "timeout": int(_env("BRAINBOX_TIMEOUT", api.get("timeout", 30))),
    }