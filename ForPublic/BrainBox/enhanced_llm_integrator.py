"""
Enhanced LLM Integration - LOCAL ONLY
====================================

SIMPLIFIED: Only calls local API at http://10.14.0.2:1234
No routing, no model selection, no cloud fallbacks.
Just local API calls.
"""

import requests
import json
from typing import Dict, Optional

class EnhancedLLMIntegrator:
    """Ultra-simple local-only LLM integrator"""
    
    def __init__(self, model_name=None):
        self.local_api_url = "http://10.14.0.2:1234"
        
        # Available models (NOT the thinking one lol)
        self.models = {
            "coder": "qwen3-coder-30b-a3b-instruct",  # Big coding model
            "fast": "qwen3-4b-instruct-2507",          # Small fast model
            # NOT using "qwen/qwen3-4b-thinking-2507" - it overthinks!
        }
        
        # Default to coder, but allow switching
        if model_name and model_name in self.models:
            self.current_model = self.models[model_name]
        else:
            self.current_model = self.models["coder"]  # Default
            
        print(f"[HASHBROWN DEBUG] Local-only mode - API: {self.local_api_url}")
        print(f"[HASHBROWN DEBUG] Using model: {self.current_model}")
    
    def process_query(self, query: str, mode: str = "auto") -> Dict:
        """Process query through local API ONLY"""
        
        # Check for model switching in query
        if query.lower().startswith("@fast "):
            self.switch_model("fast")
            query = query[6:]  # Remove "@fast " prefix
        elif query.lower().startswith("@coder "):
            self.switch_model("coder")
            query = query[7:]  # Remove "@coder " prefix
            
        print(f"[HASHBROWN DEBUG] Processing query through LOCAL API ONLY")
        
        try:
            # Call local API
            payload = {
                "model": self.current_model,  # Use selected model
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            print(f"[HASHBROWN DEBUG] Calling {self.local_api_url}/v1/chat/completions")
            
            response = requests.post(
                f"{self.local_api_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
                
                print(f"[HASHBROWN DEBUG] LOCAL API SUCCESS")
                
                return {
                    'response': content,
                    'model_used': 'local',
                    'api_endpoint': self.local_api_url,
                    'success': True
                }
            else:
                print(f"[HASHBROWN DEBUG] LOCAL API ERROR - Status: {response.status_code}")
                print(f"[HASHBROWN DEBUG] Error details: {response.text}")
                
                return {
                    'response': f'Local API error: {response.status_code} - {response.text}',
                    'model_used': 'error',
                    'api_endpoint': self.local_api_url,
                    'success': False
                }
                        
        except Exception as e:
            print(f"[HASHBROWN DEBUG] LOCAL API EXCEPTION: {e}")
            return {
                'response': f'Local API connection failed: {e}',
                'model_used': 'error',
                'api_endpoint': self.local_api_url,
                'success': False
            }
    
    def switch_model(self, model_name: str):
        """Switch between available models"""
        if model_name in self.models:
            self.current_model = self.models[model_name]
            print(f"[HASHBROWN DEBUG] Switched to model: {self.current_model}")
        else:
            print(f"[HASHBROWN DEBUG] Unknown model: {model_name}, keeping {self.current_model}")
    
    def process_query_sync(self, query: str, mode: str = "auto") -> Dict:
        """Synchronous wrapper (already sync now)"""
        return self.process_query(query, mode)
    
    def generate_response(self, query: str, processing_context: Dict, mode: str = "auto") -> Dict:
        """Generate response using local API - compatibility method for unified_brainbox"""
        print(f"[HASHBROWN DEBUG] generate_response called with mode: {mode}")
        
        # Call the local API
        result = self.process_query(query, mode)
        
        # Return in expected format
        return {
            "response": result['response'],
            "selected_model": result['model_used'],
            "routing_reason": f"LOCAL-FIRST: {result['api_endpoint']}",
            "routing_confidence": 1.0 if result['success'] else 0.0,
            "success": result['success']
        }

# Test function
if __name__ == "__main__":
    print("Testing Enhanced LLM Integrator - LOCAL ONLY")
    integrator = EnhancedLLMIntegrator()
    
    test_query = "Hello, are you working?"
    result = integrator.process_query_sync(test_query)
    
    print(f"Query: {test_query}")
    print(f"Response: {result['response']}")
    print(f"Model: {result['model_used']}")
    print(f"Endpoint: {result['api_endpoint']}")
    print(f"Success: {result['success']}")