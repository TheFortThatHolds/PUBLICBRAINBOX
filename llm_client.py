import httpx
import json

class LLMClient:
    def __init__(self, base_url, api_key="", timeout=30):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout = timeout

    def list_models(self):
        url = f"{self.base_url}/models"
        try:
            with httpx.Client(timeout=self.timeout) as s:
                r = s.get(url, headers=self.headers)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            print(f"[WARNING] Could not list models from {url}: {e}")
            return {"data": []}

    def chat(self, model, messages, **kwargs):
        url = f"{self.base_url}/chat/completions"
        payload = {"model": model, "messages": messages}
        payload.update(kwargs)
        
        try:
            with httpx.Client(timeout=self.timeout) as s:
                r = s.post(url, headers=self.headers, json=payload)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            print(f"[ERROR] Chat request failed: {e}")
            # Return placeholder response to keep system running
            return {
                "choices": [{
                    "message": {
                        "content": f"[PLACEHOLDER] Chat unavailable: {str(e)[:100]}"
                    }
                }]
            }

    def embed(self, model, inputs):
        url = f"{self.base_url}/embeddings"
        payload = {"model": model, "input": inputs}
        
        try:
            with httpx.Client(timeout=self.timeout) as s:
                r = s.post(url, headers=self.headers, json=payload)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            print(f"[ERROR] Embedding request failed: {e}")
            # Return placeholder embeddings to keep system running
            num_inputs = len(inputs) if isinstance(inputs, list) else 1
            return {
                "data": [{"embedding": [0.0] * 384} for _ in range(num_inputs)]
            }