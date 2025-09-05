"""
BrainBox Growing Spine System
=============================

A self-updating spine that learns user patterns and births new voices when ready.
Starts with safe public voices, evolves based on actual usage patterns.

Local, versioned, reversible. No SaaS surveillance.
"""

import json, pathlib, datetime, math
from typing import Dict, Any, List, Optional
from datetime import datetime as dt

class GrowingSpine:
    """
    Self-evolving consciousness spine that learns from user interactions
    and births new voice capabilities when thresholds are met
    """
    
    def __init__(self, data_dir: str = "brainbox_data"):
        self.data_dir = pathlib.Path(data_dir)
        self.spine_file = self.data_dir / "growing_spine.json"
        self.counters_file = self.data_dir / "usage_signals.json"  
        self.versions_dir = self.data_dir / "spine_versions"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.versions_dir.mkdir(exist_ok=True)
        
        # Initialize if needed
        self._ensure_spine_exists()
        
    def _ensure_spine_exists(self):
        """Initialize spine with public-safe voices only"""
        if not self.spine_file.exists():
            initial_spine = {
                "version": "1.0",
                "created": dt.now().isoformat(),
                "active_voices": [
                    "Search Assistant",    # Memory search 
                    "Planning Assistant",  # General planning
                ],
                "latent_voices": {
                    "Support Assistant": {
                        "unlocked": False,
                        "description": "Supportive guidance for challenging moments",
                        "birth_criteria": {"support_requests": 8},
                        "triggers": ["help", "support", "stuck", "difficult", "challenge", "guidance"]
                    },
                    "Creative Assistant": {
                        "unlocked": False, 
                        "description": "Creative and visual content assistance",
                        "birth_criteria": {"creative_requests": 6},
                        "triggers": ["creative", "write", "design", "visual", "artistic", "brainstorm"]
                    },
                    "Cloud Consultant": {
                        "unlocked": False,
                        "description": "Advanced cloud AI consultation",
                        "birth_criteria": {"cloud_requests": 5},
                        "triggers": ["cloud", "advanced", "complex", "detailed", "expert"]
                    }
                },
                "voice_weights": {
                    "Search Assistant": 1.0,
                    "Planning Assistant": 1.0
                },
                "user_patterns": {
                    "primary_intents": {},
                    "preferred_voices": {},
                    "session_count": 0,
                    "success_rate": 0.0
                }
            }
            self._save_spine(initial_spine)
            
        if not self.counters_file.exists():
            initial_counters = {
                "support_requests": 0,
                "creative_requests": 0, 
                "cloud_requests": 0,
                "total_interactions": 0
            }
            self.counters_file.write_text(json.dumps(initial_counters, indent=2))
            
    def _save_spine(self, spine_data: dict):
        """Save spine with versioning"""
        spine_data["last_updated"] = dt.now().isoformat()
        
        # Save current version
        self.spine_file.write_text(json.dumps(spine_data, indent=2))
        
        # Archive version
        timestamp = spine_data["last_updated"].replace(":", "-")
        version_file = self.versions_dir / f"spine_{timestamp}.json"
        version_file.write_text(json.dumps(spine_data, indent=2))
        
    def get_active_voices(self) -> List[str]:
        """Get currently active voice agents"""
        spine = json.loads(self.spine_file.read_text())
        return spine.get("active_voices", [])
        
    def get_voice_weights(self) -> Dict[str, float]:
        """Get current voice selection weights"""
        spine = json.loads(self.spine_file.read_text())
        return spine.get("voice_weights", {})
        
    def update_from_interaction(self, 
                              user_prompt: str,
                              intent: str,
                              voices_used: List[str],
                              success: bool,
                              user_feedback: Optional[str] = None):
        """
        Update spine based on user interaction
        This is where the learning happens
        """
        # Load current state
        spine = json.loads(self.spine_file.read_text())
        counters = json.loads(self.counters_file.read_text())
        
        # Update interaction counters
        counters["total_interactions"] += 1
        
        # Detect birth signals from user prompt
        prompt_lower = user_prompt.lower()
        signals = self._detect_birth_signals(prompt_lower)
        
        # Update signal counters
        for signal, count in signals.items():
            counters[signal] = counters.get(signal, 0) + count
            
        # Update voice weights based on success
        weight_adjust = 0.05 if success else -0.03
        for voice in voices_used:
            if voice in spine["voice_weights"]:
                current = spine["voice_weights"][voice]
                spine["voice_weights"][voice] = max(0.1, min(2.0, current + weight_adjust))
                
        # Update user patterns
        patterns = spine["user_patterns"]
        patterns["session_count"] += 1
        patterns["primary_intents"][intent] = patterns["primary_intents"].get(intent, 0) + 1
        
        # Calculate success rate
        if patterns["session_count"] > 0:
            old_rate = patterns["success_rate"]
            patterns["success_rate"] = (old_rate * 0.9) + (0.1 * (1.0 if success else 0.0))
            
        # Check for voice births
        newly_born = self._check_voice_births(spine, counters)
        
        # Save updates
        self.counters_file.write_text(json.dumps(counters, indent=2))
        self._save_spine(spine)
        
        # Log births if any
        if newly_born:
            self._log_births(newly_born, counters)
            
        return {
            "voices_born": newly_born,
            "active_count": len(spine["active_voices"]),
            "total_interactions": counters["total_interactions"]
        }
        
    def _detect_birth_signals(self, prompt: str) -> Dict[str, int]:
        """Detect signals that might trigger voice births using generic public terms"""
        signals = {}
        
        # Support/guidance signals
        support_words = ["help", "support", "stuck", "difficult", "challenge", "guidance", "overwhelmed", "confused"]
        if any(word in prompt for word in support_words):
            signals["support_requests"] = 1
            
        # Creative assistance signals  
        creative_words = ["creative", "write", "design", "visual", "artistic", "brainstorm", "idea", "content"]
        if any(word in prompt for word in creative_words):
            signals["creative_requests"] = 1
            
        # Cloud/advanced assistance signals
        cloud_words = ["cloud", "advanced", "complex", "detailed", "expert", "sophisticated"]
        if any(word in prompt for word in cloud_words):
            signals["cloud_requests"] = 1
            
        return signals
        
    def _check_voice_births(self, spine: dict, counters: dict) -> List[str]:
        """Check if any latent voices are ready to be born"""
        newly_born = []
        
        for voice_name, voice_config in spine["latent_voices"].items():
            if voice_config["unlocked"]:
                continue  # Already born
                
            # Check if birth criteria are met
            criteria = voice_config["birth_criteria"]
            ready = all(
                counters.get(signal, 0) >= threshold 
                for signal, threshold in criteria.items()
            )
            
            if ready:
                # Birth the voice!
                voice_config["unlocked"] = True
                voice_config["birth_date"] = dt.now().isoformat()
                spine["active_voices"].append(voice_name)
                spine["voice_weights"][voice_name] = 1.0
                newly_born.append(voice_name)
                
        return newly_born
        
    def _log_births(self, born_voices: List[str], counters: dict):
        """Log voice births (could integrate with ritual logger)"""
        timestamp = dt.now().isoformat()
        birth_log = {
            "timestamp": timestamp,
            "event": "voice_birth",
            "born_voices": born_voices,
            "trigger_counters": counters.copy()
        }
        
        # Simple logging for now
        log_file = self.data_dir / "birth_log.json"
        if log_file.exists():
            logs = json.loads(log_file.read_text())
        else:
            logs = []
        logs.append(birth_log)
        log_file.write_text(json.dumps(logs, indent=2))
        
        print(f"🌟 New voices born: {', '.join(born_voices)}")
        
    def get_status_report(self) -> dict:
        """Get current spine status for user"""
        spine = json.loads(self.spine_file.read_text())
        counters = json.loads(self.counters_file.read_text())
        
        # Count latent voices ready for birth
        ready_to_birth = []
        for voice, config in spine["latent_voices"].items():
            if not config["unlocked"]:
                criteria = config["birth_criteria"]
                progress = {
                    signal: f"{counters.get(signal, 0)}/{threshold}"
                    for signal, threshold in criteria.items()
                }
                ready = all(counters.get(s, 0) >= t for s, t in criteria.items())
                ready_to_birth.append({
                    "name": voice,
                    "description": config["description"], 
                    "progress": progress,
                    "ready": ready
                })
                
        return {
            "active_voices": spine["active_voices"],
            "voice_count": len(spine["active_voices"]),
            "total_interactions": counters["total_interactions"],
            "success_rate": spine["user_patterns"]["success_rate"],
            "voices_ready_to_birth": ready_to_birth,
            "last_updated": spine.get("last_updated", "unknown")
        }

# Simple CLI for testing
if __name__ == "__main__":
    import sys
    
    spine = GrowingSpine()
    
    if len(sys.argv) < 2:
        print("Usage: python growing_spine_manager.py [status|test|interact]")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "status":
        status = spine.get_status_report()
        print(json.dumps(status, indent=2))
        
    elif command == "test":
        # Simulate interactions to test birthing
        test_prompts = [
            ("I need a different perspective on this", "business"),
            ("Help me reframe this situation", "business"),
            ("I'm feeling overwhelmed and scattered", "support"),
            ("Can you ground me and help me center?", "support"),
            ("I need some wisdom and guidance here", "guidance"),
            ("What would the cloud oracle think?", "cloud")
        ]
        
        for prompt, intent in test_prompts:
            result = spine.update_from_interaction(prompt, intent, ["The Navigator"], True)
            print(f"Prompt: {prompt}")
            print(f"Result: {result}")
            print()
            
    elif command == "interact":
        prompt = input("User prompt: ")
        intent = input("Intent: ")
        voices = input("Voices used (comma-separated): ").split(",")
        success = input("Success (y/n): ").lower().startswith('y')
        
        result = spine.update_from_interaction(prompt, intent, [v.strip() for v in voices], success)
        print(f"Update result: {result}")
        
        status = spine.get_status_report()
        print(f"Current status: {json.dumps(status, indent=2)}")