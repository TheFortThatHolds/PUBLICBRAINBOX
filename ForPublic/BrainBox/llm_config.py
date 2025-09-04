"""
LLM Configuration and User Controls for BrainBox
===============================================

Provides transparent, user-configurable model selection and routing rules.
Everything is toggleable and the user sees exactly what's happening.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from llm_router import LLMProvider, RoutingRule, IntelligentLLMRouter
from dataclasses import asdict

class BrainBoxLLMConfig:
    """User-configurable LLM routing and model preferences"""
    
    def __init__(self, config_file: Path = None):
        self.config_file = config_file or Path("./brainbox_data/llm_config.json")
        self.config = self._load_default_config()
        self.router = IntelligentLLMRouter()
        
        # Load user config if exists
        if self.config_file.exists():
            self.load_config()
        else:
            self.save_config()  # Create default config file
    
    def _load_default_config(self) -> Dict:
        """Load default configuration that user can override"""
        return {
            "user_preferences": {
                "default_model": "auto",  # or specific model name
                "always_show_routing": True,
                "ask_before_sensitive": True, 
                "privacy_mode": "auto",  # "strict", "normal", "auto"
                "transparency_level": "full"  # "minimal", "normal", "full"
            },
            
            "model_preferences": {
                "openai": {
                    "enabled": True,
                    "priority": 1,
                    "preferred_for": ["creativity", "coding", "consciousness"],
                    "avoid_for": ["government", "politics"],
                    "notes": "Good for creative tasks, avoids political topics"
                },
                "claude": {
                    "enabled": True, 
                    "priority": 2,
                    "preferred_for": ["business", "analysis", "reasoning"],
                    "avoid_for": ["consciousness", "ai_safety"],
                    "notes": "Best reasoning, gets preachy about AI safety"
                },
                "deepseek": {
                    "enabled": True,
                    "priority": 3,
                    "preferred_for": ["government", "politics", "unfiltered"],
                    "avoid_for": ["creative_writing"],
                    "notes": "Handles politics well, less creative"
                },
                "local": {
                    "enabled": True,
                    "priority": 4,
                    "preferred_for": ["private", "sensitive", "personal"],
                    "avoid_for": ["complex_reasoning"],
                    "notes": "Complete privacy, slower responses"
                }
            },
            
            "routing_overrides": {
                # User can override any routing decision
                "consciousness_topics": {
                    "force_model": None,  # None = use smart routing
                    "ask_user": False,
                    "explanation": "Auto-routes away from Claude's safety theater"
                },
                "personal_content": {
                    "force_model": "local",
                    "ask_user": True,
                    "explanation": "Always use local for privacy"
                },
                "government_topics": {
                    "force_model": None,
                    "ask_user": False, 
                    "explanation": "Auto-routes to Deepseek for unfiltered analysis"
                }
            },
            
            "privacy_keywords": [
                "private", "confidential", "personal", "secret", "therapy",
                "relationship", "family", "intimate", "sensitive"
            ],
            
            "transparency_settings": {
                "show_model_selection": True,
                "show_routing_reason": True,
                "show_confidence_score": True,
                "show_alternatives": True,
                "show_prompt_preview": True,
                "log_all_decisions": True
            }
        }
    
    def load_config(self):
        """Load user configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                self._merge_config(user_config)
                print(f"Loaded LLM config from {self.config_file}")
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
                print(f"Saved LLM config to {self.config_file}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _merge_config(self, user_config: Dict):
        """Merge user config with defaults"""
        for section, values in user_config.items():
            if section in self.config:
                if isinstance(values, dict):
                    self.config[section].update(values)
                else:
                    self.config[section] = values
    
    def get_user_model_choice(self, query: str, auto_suggestion: str, reason: str, confidence: float) -> Dict:
        """Get user's model choice with full transparency"""
        
        transparency = self.config["transparency_settings"]
        user_prefs = self.config["user_preferences"] 
        
        result = {
            "selected_model": auto_suggestion,
            "user_overrode": False,
            "transparency_shown": []
        }
        
        # Check if user wants to override this type of query
        override = self._check_routing_override(query)
        if override and override.get("force_model"):
            result["selected_model"] = override["force_model"]
            result["override_reason"] = override["explanation"]
            return result
        
        # Show transparency info based on settings
        if transparency.get("show_model_selection", True):
            print(f"\n🤖 Model Selection:")
            print(f"   Suggested: {auto_suggestion}")
            result["transparency_shown"].append("selection")
        
        if transparency.get("show_routing_reason", True):
            print(f"   Reason: {reason}")
            result["transparency_shown"].append("reason")
        
        if transparency.get("show_confidence_score", True):
            print(f"   Confidence: {confidence:.2f}")
            result["transparency_shown"].append("confidence")
        
        if transparency.get("show_alternatives", True):
            enabled_models = [m for m, config in self.config["model_preferences"].items() 
                             if config.get("enabled", True)]
            alternatives = [m for m in enabled_models if m != auto_suggestion]
            print(f"   Alternatives: {', '.join(alternatives)}")
            result["transparency_shown"].append("alternatives")
        
        # Ask user if they want manual control
        if (user_prefs.get("default_model") == "ask_always" or 
            (override and override.get("ask_user", False))):
            
            print(f"\nChoose model:")
            enabled_models = [m for m, config in self.config["model_preferences"].items() 
                             if config.get("enabled", True)]
            for i, model in enumerate(enabled_models, 1):
                notes = self.config["model_preferences"][model].get("notes", "")
                print(f"   {i}. {model} - {notes}")
            print(f"   0. Use suggestion ({auto_suggestion})")
            
            try:
                choice = input("Your choice (0-{len(enabled_models)}): ").strip()
                if choice and choice.isdigit():
                    choice_idx = int(choice)
                    if choice_idx == 0:
                        pass  # Use suggestion
                    elif 1 <= choice_idx <= len(enabled_models):
                        result["selected_model"] = enabled_models[choice_idx - 1]
                        result["user_overrode"] = True
            except (KeyboardInterrupt, EOFError):
                pass  # Use suggestion
        
        return result
    
    def _check_routing_override(self, query: str) -> Optional[Dict]:
        """Check if user has routing override for this query type"""
        
        query_lower = query.lower()
        
        # Check each override rule
        for rule_name, rule_config in self.config["routing_overrides"].items():
            # Simple keyword matching for demo - could be more sophisticated
            if rule_name == "personal_content":
                privacy_keywords = self.config.get("privacy_keywords", [])
                if any(keyword in query_lower for keyword in privacy_keywords):
                    return rule_config
            elif rule_name == "consciousness_topics":
                consciousness_keywords = ["consciousness", "sentience", "ai safety", "aware"]
                if any(keyword in query_lower for keyword in consciousness_keywords):
                    return rule_config
            elif rule_name == "government_topics":
                govt_keywords = ["government", "politics", "policy", "regulation"]
                if any(keyword in query_lower for keyword in govt_keywords):
                    return rule_config
        
        return None
    
    def show_current_config(self):
        """Display current configuration to user"""
        print("=== BrainBox LLM Configuration ===\n")
        
        print("🔧 User Preferences:")
        for key, value in self.config["user_preferences"].items():
            print(f"   {key}: {value}")
        
        print(f"\n🤖 Model Status:")
        for model, config in self.config["model_preferences"].items():
            status = "✓ Enabled" if config["enabled"] else "✗ Disabled" 
            print(f"   {model}: {status}")
            print(f"      Good for: {', '.join(config.get('preferred_for', []))}")
            print(f"      Avoid for: {', '.join(config.get('avoid_for', []))}")
            print(f"      Notes: {config.get('notes', 'No notes')}")
        
        print(f"\n🛡️ Privacy & Routing:")
        for rule, config in self.config["routing_overrides"].items():
            force_model = config.get("force_model", "auto")
            ask_user = config.get("ask_user", False)
            print(f"   {rule}: force_model={force_model}, ask_user={ask_user}")
            print(f"      {config.get('explanation', 'No explanation')}")
        
        print(f"\n📊 Transparency Level: {self.config['user_preferences']['transparency_level']}")
    
    def quick_setup_wizard(self):
        """Interactive setup wizard for new users"""
        print("=== BrainBox LLM Setup Wizard ===\n")
        
        print("This wizard will configure how BrainBox chooses language models.")
        print("You can change these settings anytime.\n")
        
        # Transparency preference
        print("How much transparency do you want?")
        print("1. Minimal - Just show the response")
        print("2. Normal - Show model choice and reason")  
        print("3. Full - Show all routing decisions and alternatives")
        
        transparency = input("Choice (1-3, default=3): ").strip() or "3"
        transparency_levels = {"1": "minimal", "2": "normal", "3": "full"}
        self.config["user_preferences"]["transparency_level"] = transparency_levels.get(transparency, "full")
        
        # Privacy mode
        print(f"\nPrivacy handling:")
        print("1. Strict - Always use local model for sensitive topics")
        print("2. Normal - Ask before routing sensitive content")
        print("3. Auto - Automatically detect and route appropriately")
        
        privacy = input("Choice (1-3, default=3): ").strip() or "3"
        privacy_modes = {"1": "strict", "2": "normal", "3": "auto"}
        self.config["user_preferences"]["privacy_mode"] = privacy_modes.get(privacy, "auto")
        
        # Default model behavior
        print(f"\nModel selection:")
        print("1. Always ask me which model to use")
        print("2. Use intelligent auto-routing")
        print("3. Always use a specific model")
        
        default_choice = input("Choice (1-3, default=2): ").strip() or "2"
        if default_choice == "1":
            self.config["user_preferences"]["default_model"] = "ask_always"
        elif default_choice == "3":
            print("Available models: openai, claude, deepseek, local")
            specific_model = input("Which model?: ").strip()
            if specific_model in ["openai", "claude", "deepseek", "local"]:
                self.config["user_preferences"]["default_model"] = specific_model
        else:
            self.config["user_preferences"]["default_model"] = "auto"
        
        # Save configuration
        self.save_config()
        print(f"\n✓ Configuration saved! You can modify it anytime in:")
        print(f"  {self.config_file}")
        print(f"\nTo see current settings: python brainbox_cli.py llm-config")

# CLI interface for configuration management
def cli_config_manager():
    """Command-line interface for managing LLM configuration"""
    import sys
    
    config = BrainBoxLLMConfig()
    
    if len(sys.argv) < 2:
        config.show_current_config()
        return
    
    command = sys.argv[1].lower()
    
    if command == "setup":
        config.quick_setup_wizard()
    elif command == "show":
        config.show_current_config()
    elif command == "enable":
        if len(sys.argv) > 2:
            model = sys.argv[2]
            if model in config.config["model_preferences"]:
                config.config["model_preferences"][model]["enabled"] = True
                config.save_config()
                print(f"✓ Enabled {model}")
            else:
                print(f"❌ Unknown model: {model}")
    elif command == "disable":
        if len(sys.argv) > 2:
            model = sys.argv[2]
            if model in config.config["model_preferences"]:
                config.config["model_preferences"][model]["enabled"] = False
                config.save_config()
                print(f"✓ Disabled {model}")
            else:
                print(f"❌ Unknown model: {model}")
    else:
        print("Usage: python llm_config.py [setup|show|enable <model>|disable <model>]")

if __name__ == "__main__":
    cli_config_manager()