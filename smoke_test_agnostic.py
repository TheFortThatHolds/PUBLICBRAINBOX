"""
BrainBox Model-Agnostic Architecture Smoke Test
===============================================

Tests that the refactored system actually works without provider coupling.
"""

import sys
import pathlib
import subprocess
import importlib.util

def test_no_provider_coupling():
    """Ensure core files don't import provider-specific SDKs"""
    print("[TEST] Checking for provider coupling...")
    
    core_files = [
        "unified_brainbox.py",
        "config_loader.py", 
        "llm_client.py",
        "model_select.py",
        "llm_init.py"
    ]
    
    bad_imports = ["import openai", "import anthropic", "import groq", "from openai", "from anthropic", "from groq"]
    
    for file_name in core_files:
        file_path = pathlib.Path(file_name)
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            for bad_import in bad_imports:
                if bad_import in content:
                    print(f"[FAIL] {file_name} contains provider coupling: {bad_import}")
                    return False
    
    print("[PASS] No provider coupling found in core files")
    return True

def test_config_loading():
    """Test config loading works"""
    print("[TEST] Testing config loading...")
    
    try:
        from config_loader import load_config
        cfg = load_config()
        required_keys = ["base_url", "chat_model", "embed_model", "api_key", "timeout"]
        
        for key in required_keys:
            if key not in cfg:
                print(f"[FAIL] Missing config key: {key}")
                return False
                
        print(f"[PASS] Config loaded: {cfg['base_url']}")
        return True
    except Exception as e:
        print(f"[FAIL] Config loading failed: {e}")
        return False

def test_llm_client():
    """Test LLM client can be instantiated"""
    print("[TEST] Testing LLM client...")
    
    try:
        from llm_client import LLMClient
        client = LLMClient("http://localhost:1234/v1")
        
        # Test that it has the expected methods
        methods = ["list_models", "chat", "embed"]
        for method in methods:
            if not hasattr(client, method):
                print(f"[FAIL] LLMClient missing method: {method}")
                return False
        
        print("[PASS] LLM client instantiated successfully")
        return True
    except Exception as e:
        print(f"[FAIL] LLM client test failed: {e}")
        return False

def test_model_selection():
    """Test model selection logic"""
    print("[TEST] Testing model selection...")
    
    try:
        from model_select import pick_models
        
        # Test with mock models list
        mock_models = [
            {"id": "gpt-3.5-turbo"},
            {"id": "text-embedding-ada-002"},
            {"id": "llama-2-7b-chat"},
            {"id": "nomic-embed-text-v1"}
        ]
        
        chat, embed = pick_models(mock_models)
        
        if not chat:
            print("[FAIL] No chat model selected")
            return False
            
        if not embed:
            print("[FAIL] No embed model selected")
            return False
            
        print(f"[PASS] Selected chat: {chat}, embed: {embed}")
        return True
    except Exception as e:
        print(f"[FAIL] Model selection test failed: {e}")
        return False

def test_llm_init():
    """Test LLM initialization"""
    print("[TEST] Testing LLM initialization...")
    
    try:
        from llm_init import init_llm
        
        # This should work even if no server is running (graceful degradation)
        client, chat_model, embed_model = init_llm()
        
        if not client:
            print("[FAIL] No client returned")
            return False
            
        print(f"[PASS] LLM init successful: {client.base_url}")
        return True
    except Exception as e:
        # Expected to fail if no models configured - that's OK for smoke test
        if "Model configuration incomplete" in str(e):
            print("[PASS] LLM init failed gracefully (no models configured)")
            return True
        else:
            print(f"[FAIL] LLM init failed unexpectedly: {e}")
            return False

def test_legacy_compatibility():
    """Test legacy imports still work"""
    print("[TEST] Testing legacy compatibility...")
    
    try:
        from llm_config import BrainBoxLLMConfig
        config = BrainBoxLLMConfig()
        
        # Should have the expected attributes
        attrs = ["base_url", "chat_model", "embed_model", "api_key", "timeout"]
        for attr in attrs:
            if not hasattr(config, attr):
                print(f"[FAIL] Legacy config missing attribute: {attr}")
                return False
                
        print("[PASS] Legacy compatibility maintained")
        return True
    except Exception as e:
        print(f"[FAIL] Legacy compatibility test failed: {e}")
        return False

def run_smoke_tests():
    """Run all smoke tests"""
    print("BrainBox Model-Agnostic Architecture Smoke Test")
    print("=" * 50)
    
    tests = [
        test_no_provider_coupling,
        test_config_loading,
        test_llm_client,
        test_model_selection,
        test_llm_init,
        test_legacy_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Architecture is model-agnostic.")
        return True
    else:
        print("[FAILURE] Some tests failed. Architecture needs work.")
        return False

if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)