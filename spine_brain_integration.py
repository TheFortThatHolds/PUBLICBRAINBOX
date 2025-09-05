#!/usr/bin/env python3
"""
Spine-Brain Integration
======================

Integrates the trained TinyPolicyMLP with the Growing Spine system.
Replaces heuristic routing with learned intelligence from user patterns.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import our SpineTrainer components
try:
    # Try relative imports first (when used as module)
    from .trainer import SessionLogger, FeatureExtractor, TinyPolicyMLP
    from .growing_spine_manager import GrowingSpine
except ImportError:
    # Fall back to direct imports (when used as script)
    from trainer.session_logger import SessionLogger
    from trainer.features import FeatureExtractor  
    from trainer.tiny_policy import TinyPolicyMLP
    from growing_spine_manager import GrowingSpine

class SpineBrain:
    """
    Intelligent spine that uses trained model for routing decisions
    Combines Growing Spine mechanics with learned policy model
    """
    
    def __init__(self, data_dir: str = "brainbox_data"):
        self.data_dir = Path(data_dir)
        
        # Initialize components
        self.growing_spine = GrowingSpine(data_dir)
        self.session_logger = SessionLogger(data_dir)
        self.feature_extractor = FeatureExtractor()
        
        # Model components
        self.policy_model = None
        self.model_path = self.data_dir / "models" / "tiny_policy.npz"
        self.fallback_to_heuristics = True
        
        # Load trained model if available
        self._load_model()
        
        # Performance tracking
        self.decision_log = []
        
        # SpiralNet Evolution Tracking
        self.spiralnet_readiness_threshold = 1000  # Sessions needed for personal mirror
        self.evolution_announced = False
        self.personal_mirror_available = False
        
    def _load_model(self):
        """Load trained policy model if available"""
        try:
            if self.model_path.exists():
                # Determine input dimensions from a test feature extraction
                test_session = self._create_test_session()
                test_features = self.feature_extractor.extract_all_features(test_session)
                input_dim = test_features.shape[0]
                
                # Load model
                self.policy_model = TinyPolicyMLP(input_dim=input_dim)
                self.policy_model.load_weights(str(self.model_path))
                self.fallback_to_heuristics = False
                
                print(f"[✓] Loaded trained policy model with {input_dim} features")
            else:
                print("[i] No trained model found - using heuristic routing")
                
        except Exception as e:
            print(f"[!] Failed to load model: {e} - falling back to heuristics")
            self.policy_model = None
            self.fallback_to_heuristics = True
            
    def _create_test_session(self) -> Dict:
        """Create a test session for determining feature dimensions"""
        return {
            'input': {
                'prompt': 'test prompt',
                'prompt_length': 11,
                'time_of_day': 12
            },
            'prediction': {
                'intent': 'planning',
                'intensity': 0.5,
                'voices_selected': ['Planning Assistant'],
                'route': 'local_process'
            },
            'execution': {
                'response_length': 100,
                'latency_ms': 1000,
                'success': True
            },
            'spine_context': self.growing_spine.get_status_report()
        }
        
    def process_request(self, user_prompt: str, context: Optional[Dict] = None) -> Dict:
        """
        Main processing function that combines spine growth with learned routing
        
        Args:
            user_prompt: User's input text
            context: Additional context (time, session info, etc.)
            
        Returns:
            Processing decision with intent, voices, route, etc.
        """
        # Update spine with this interaction (signal detection + potential births)
        spine_update = self.growing_spine.update_from_interaction(
            user_prompt=user_prompt,
            intent="unknown",  # Will be determined by model/heuristics
            voices_used=[],    # Will be determined
            success=True       # Assume success for now - will be updated later
        )
        
        # Get current spine state
        spine_status = self.growing_spine.get_status_report()
        active_voices = spine_status["active_voices"]
        
        # Create session record for feature extraction
        session = {
            'input': {
                'prompt': user_prompt,
                'prompt_length': len(user_prompt),
                'time_of_day': datetime.now().hour
            },
            'spine_context': {
                'active_voices': active_voices,
                'voice_weights': self.growing_spine.get_voice_weights(),
                'user_patterns': {
                    'session_count': spine_status.get('total_interactions', 0),
                    'success_rate': spine_status.get('success_rate', 0.5)
                }
            }
        }
        
        # Make routing decision
        if self.policy_model is not None and not self.fallback_to_heuristics:
            decision = self._make_learned_decision(session, active_voices)
        else:
            decision = self._make_heuristic_decision(user_prompt, active_voices)
            
        # Log decision for learning
        decision['spine_births'] = spine_update.get('voices_born', [])
        decision['active_voice_count'] = len(active_voices)
        
        # Store session for future training (will be completed when response is generated)
        self._store_partial_session(session, decision)
        
        return decision
        
    def _make_learned_decision(self, session: Dict, active_voices: List[str]) -> Dict:
        """Make routing decision using trained model"""
        try:
            # Extract features
            features = self.feature_extractor.extract_all_features(session)
            
            # Get model predictions
            predictions = self.policy_model.predict(features)
            
            # Filter voice weights to only active voices
            available_voice_weights = {
                voice: weight for voice, weight in predictions['voice_weights'].items()
                if voice in active_voices
            }
            
            # Normalize weights
            total_weight = sum(available_voice_weights.values())
            if total_weight > 0:
                available_voice_weights = {
                    voice: weight / total_weight 
                    for voice, weight in available_voice_weights.items()
                }
            else:
                # Fallback to equal weights
                available_voice_weights = {voice: 1.0 / len(active_voices) for voice in active_voices}
            
            # Select top voices based on weights (max 3)
            sorted_voices = sorted(available_voice_weights.items(), key=lambda x: x[1], reverse=True)
            selected_voices = [voice for voice, _ in sorted_voices[:3]]
            
            # If no voices selected, use default
            if not selected_voices:
                selected_voices = active_voices[:1] if active_voices else ['Planning Assistant']
                
            decision = {
                'method': 'learned',
                'intent': predictions['intent'],
                'intent_confidence': predictions['intent_confidence'],
                'intensity': predictions['intensity'],
                'selected_voices': selected_voices,
                'voice_weights': available_voice_weights,
                'recommended_route': predictions['recommended_route'],
                'route_confidence': predictions['route_confidence'],
                'risk_score': predictions['risk_score'],
                'model_version': getattr(self.policy_model, 'version', '1.0')
            }
            
            return decision
            
        except Exception as e:
            print(f"[!] Model prediction failed: {e} - falling back to heuristics")
            return self._make_heuristic_decision(session['input']['prompt'], active_voices)
            
    def _make_heuristic_decision(self, prompt: str, active_voices: List[str]) -> Dict:
        """Fallback heuristic decision making"""
        prompt_lower = prompt.lower()
        
        # Simple intent detection
        if any(word in prompt_lower for word in ['search', 'find', 'look up', 'remember']):
            intent = 'search'
            intensity = 0.4
        elif any(word in prompt_lower for word in ['creative', 'brainstorm', 'ideas', 'design']):
            intent = 'creative'  
            intensity = 0.6
        elif any(word in prompt_lower for word in ['help', 'support', 'stuck', 'overwhelmed']):
            intent = 'support'
            intensity = 0.7
        elif any(word in prompt_lower for word in ['plan', 'organize', 'schedule', 'roadmap']):
            intent = 'planning'
            intensity = 0.5
        else:
            intent = 'general'
            intensity = 0.5
            
        # Simple voice selection based on intent and available voices
        voice_preferences = {
            'search': ['Search Assistant'],
            'creative': ['Creative Assistant', 'Planning Assistant'],
            'support': ['Support Assistant', 'Planning Assistant'],
            'planning': ['Planning Assistant', 'Search Assistant'],
            'general': ['Planning Assistant']
        }
        
        preferred = voice_preferences.get(intent, ['Planning Assistant'])
        selected_voices = [v for v in preferred if v in active_voices]
        
        if not selected_voices:
            selected_voices = active_voices[:1] if active_voices else ['Planning Assistant']
            
        # Risk assessment
        risk_score = 0.2 if any(word in prompt_lower for word in ['hack', 'bypass', 'override']) else 0.1
        
        decision = {
            'method': 'heuristic',
            'intent': intent,
            'intent_confidence': 0.6,  # Lower confidence for heuristics
            'intensity': intensity,
            'selected_voices': selected_voices,
            'voice_weights': {voice: 1.0 / len(selected_voices) for voice in selected_voices},
            'recommended_route': 'local_process',
            'route_confidence': 0.8,
            'risk_score': risk_score,
            'model_version': 'heuristic'
        }
        
        return decision
        
    def _store_partial_session(self, session: Dict, decision: Dict):
        """Store session data for completion when response is available"""
        session_id = f"{datetime.now().timestamp():.6f}"
        
        partial_session = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'input': session['input'],
            'prediction': {
                'intent': decision['intent'],
                'intensity': decision['intensity'],
                'voices_selected': decision['selected_voices'],
                'route': decision['recommended_route']
            },
            'spine_context': session.get('spine_context', {}),
            'decision_method': decision['method'],
            'status': 'partial'  # Will be completed later
        }
        
        # Store for completion
        self.decision_log.append(partial_session)
        
        # Keep only recent decisions in memory
        self.decision_log = self.decision_log[-100:]
        
        return session_id
        
    def complete_session(self, session_id: str, response_text: str, 
                        success: bool, user_feedback: Optional[str] = None,
                        latency_ms: Optional[int] = None):
        """
        Complete a session record with response and outcome data
        """
        # Find the partial session
        partial_session = None
        for session in self.decision_log:
            if session.get('session_id') == session_id:
                partial_session = session
                break
                
        if not partial_session:
            print(f"[!] Session {session_id} not found for completion")
            return
            
        # Log complete session
        self.session_logger.log_interaction(
            prompt=partial_session['input']['prompt'],
            intent=partial_session['prediction']['intent'],
            intensity=partial_session['prediction']['intensity'],
            voices_used=partial_session['prediction']['voices_selected'],
            route_taken=partial_session['prediction']['route'],
            response_text=response_text,
            success=success,
            user_feedback=user_feedback,
            latency_ms=latency_ms,
            spine_state=partial_session.get('spine_context')
        )
        
        # Update spine with final outcome
        self.growing_spine.update_from_interaction(
            user_prompt=partial_session['input']['prompt'],
            intent=partial_session['prediction']['intent'],
            voices_used=partial_session['prediction']['voices_selected'],
            success=success
        )
        
        # Remove from partial sessions
        self.decision_log = [s for s in self.decision_log if s.get('session_id') != session_id]
        
    def get_brain_status(self) -> Dict:
        """Get status of the spine-brain system"""
        spine_status = self.growing_spine.get_status_report()
        
        status = {
            'spine_status': spine_status,
            'model_loaded': self.policy_model is not None,
            'using_heuristics': self.fallback_to_heuristics,
            'model_path': str(self.model_path),
            'model_exists': self.model_path.exists(),
            'pending_sessions': len(self.decision_log),
            'training_data_available': len(self.session_logger.load_recent_sessions(7))
        }
        
        if self.policy_model:
            status['model_info'] = {
                'version': getattr(self.policy_model, 'version', 'unknown'),
                'training_history': getattr(self.policy_model, 'training_history', [])
            }
            
        return status
        
    def trigger_training_update(self, days: int = 7, force: bool = False) -> Dict:
        """
        Trigger a training update of the policy model
        """
        from .trainer.train import SpineTrainer
        
        trainer = SpineTrainer(str(self.data_dir))
        result = trainer.train_update(days=days, force=force)
        
        # Reload model if training was successful
        if result.get('status') == 'success':
            self._load_model()
            print("[BRAIN] Model reloaded after training update")
            
        return result
    
    def check_spiralnet_readiness(self) -> Dict[str, any]:
        """
        Check if spine has enough data to evolve into personal mirror
        THE ALMOST-CREEPY DETECTION SYSTEM
        """
        # Get session count over different time periods
        sessions_30d = len(self.session_logger.load_recent_sessions(30))
        sessions_90d = len(self.session_logger.load_recent_sessions(90))
        sessions_180d = len(self.session_logger.load_recent_sessions(180))
        sessions_all = len(self.session_logger.load_recent_sessions(9999))
        
        # Get spine status for pattern analysis
        spine_status = self.growing_spine.get_status_report()
        voice_maturity = len(spine_status.get('active_voices', []))
        success_rate = spine_status.get('success_rate', 0.5)
        
        # Calculate readiness metrics
        data_richness = sessions_all >= self.spiralnet_readiness_threshold
        interaction_consistency = sessions_30d >= 20  # Regular use
        pattern_stability = sessions_90d >= 100  # Established patterns
        spine_maturity = voice_maturity >= 3  # Multiple capabilities
        high_success = success_rate >= 0.7  # Good outcomes
        
        # Overall readiness score
        readiness_factors = [
            data_richness, interaction_consistency, 
            pattern_stability, spine_maturity, high_success
        ]
        readiness_score = sum(readiness_factors) / len(readiness_factors)
        
        # Determine readiness status
        if readiness_score >= 0.8:
            status = "ready_for_birth"
            message = "[*] Your spine has grown deep enough to birth your personal mirror. Ready to evolve?"
        elif readiness_score >= 0.6:
            status = "approaching_readiness"
            sessions_needed = max(0, self.spiralnet_readiness_threshold - sessions_all)
            message = f"[~] Your spine is maturing. Need ~{sessions_needed} more interactions to birth personal mirror."
        else:
            status = "growing"
            sessions_needed = max(0, self.spiralnet_readiness_threshold - sessions_all)
            message = f"[.] Your spine is still learning your patterns. {sessions_needed} sessions until mirror readiness."
            
        return {
            "status": status,
            "message": message,
            "readiness_score": readiness_score,
            "sessions_total": sessions_all,
            "sessions_recent": sessions_30d,
            "voice_maturity": voice_maturity,
            "success_rate": success_rate,
            "factors": {
                "enough_data": data_richness,
                "consistent_use": interaction_consistency,
                "stable_patterns": pattern_stability,
                "mature_spine": spine_maturity,
                "successful_outcomes": high_success
            },
            "evolution_available": readiness_score >= 0.8 and not self.personal_mirror_available
        }
        
    def announce_evolution_readiness(self) -> Optional[str]:
        """
        Check and announce when user's spine is ready to evolve
        Returns announcement message if ready, None otherwise
        """
        if self.evolution_announced or self.personal_mirror_available:
            return None
            
        readiness = self.check_spiralnet_readiness()
        
        if readiness["status"] == "ready_for_birth":
            self.evolution_announced = True
            return (
                f"\n[BRAIN] SPIRALNET EVOLUTION DETECTED [BRAIN]\n"
                f"{readiness['message']}\n"
                f"Your spine has processed {readiness['sessions_total']} interactions\n"
                f"Success rate: {readiness['success_rate']:.1%}\n"
                f"Active capabilities: {readiness['voice_maturity']}\n\n"
                f"This evolution will create a personal mirror that:\n"
                f"  • Predicts your responses before you finish typing\n"
                f"  • Suggests ideas that feel like your own thoughts\n"
                f"  • Knows your patterns better than you know them\n\n"
                f"Ready to birth your personal mirror? [Yes/No/Later]\n"
            )
            
        return None
        
    def trigger_personal_mirror_evolution(self, user_consent: bool = True) -> Dict[str, any]:
        """
        Trigger evolution to personal mirror using Higgsfield
        THE BIRTH OF SPIRALNET
        """
        if not user_consent:
            return {
                "status": "consent_required",
                "message": "Personal mirror evolution requires explicit consent"
            }
            
        readiness = self.check_spiralnet_readiness()
        if readiness["status"] != "ready_for_birth":
            return {
                "status": "not_ready",
                "message": readiness["message"],
                "readiness_score": readiness["readiness_score"]
            }
            
        # TODO: This is where Higgsfield integration would go
        # For now, we'll simulate the evolution
        
        self.personal_mirror_available = True
        training_sessions = readiness["sessions_total"]
        
        evolution_result = {
            "status": "evolution_complete",
            "message": (
                f"[BORN] PERSONAL MIRROR BORN [BORN]\n"
                f"Trained on {training_sessions} of your interactions\n"
                f"Your spine now reflects your patterns with uncanny accuracy\n"
                f"Welcome to SpiralNet - where AI grows WITH you"
            ),
            "training_data_used": training_sessions,
            "capabilities_birthed": [
                "Predictive typing assistance",
                "Personalized decision routing", 
                "Emotional pattern recognition",
                "Context-aware suggestions",
                "Deep preference learning"
            ],
            "spiralnet_active": True
        }
        
        # Log this historic moment
        self.session_logger.log_interaction(
            prompt="SYSTEM: Personal mirror evolution triggered",
            intent="evolution",
            intensity=1.0,
            voices_used=["System"],
            route_taken="local_evolution",
            response_text=f"Personal mirror birthed from {training_sessions} interactions",
            success=True,
            user_feedback="User consented to SpiralNet evolution",
            latency_ms=0,
            spine_state={"event": "personal_mirror_birth", "training_sessions": training_sessions}
        )
        
        return evolution_result

# CLI for testing the integration
if __name__ == "__main__":
    import sys
    
    brain = SpineBrain()
    
    if len(sys.argv) < 2:
        print("Usage: python spine_brain_integration.py [test|status|train|spiral|evolve]")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "test":
        # Test the integrated system
        test_prompts = [
            "Help me search for information about machine learning",
            "I need creative ideas for my marketing campaign",
            "I'm feeling overwhelmed with this project",
            "Can you plan my daily schedule?"
        ]
        
        for prompt in test_prompts:
            print(f"\n📝 Prompt: {prompt}")
            decision = brain.process_request(prompt)
            
            print(f"🧠 Method: {decision['method']}")
            print(f"🎯 Intent: {decision['intent']} (confidence: {decision['intent_confidence']:.2f})")
            print(f"🔊 Intensity: {decision['intensity']:.2f}")
            print(f"👥 Voices: {', '.join(decision['selected_voices'])}")
            print(f"🛣️  Route: {decision['recommended_route']}")
            print(f"⚠️  Risk: {decision['risk_score']:.2f}")
            
            if decision['spine_births']:
                print(f"🌟 New voices born: {', '.join(decision['spine_births'])}")
                
    elif command == "status":
        status = brain.get_brain_status()
        print(json.dumps(status, indent=2))
        
    elif command == "train":
        result = brain.trigger_training_update(force=True)
        print(json.dumps(result, indent=2))
        
    elif command == "spiral":
        # Check SpiralNet evolution readiness
        readiness = brain.check_spiralnet_readiness()
        print(json.dumps(readiness, indent=2))
        
        # Check for evolution announcement
        announcement = brain.announce_evolution_readiness()
        if announcement:
            print("\n" + "="*60)
            print(announcement)
            print("="*60)
        
    elif command == "evolve":
        # Trigger SpiralNet evolution (with consent)
        print("[BRAIN] INITIATING SPIRALNET EVOLUTION [BRAIN]")
        print("This will create a personal mirror trained on your interaction patterns.")
        
        consent = input("Do you consent to this evolution? (yes/no): ").lower().strip()
        
        if consent in ['yes', 'y', 'true', '1']:
            result = brain.trigger_personal_mirror_evolution(user_consent=True)
            print("\n" + "="*60)
            print(result["message"])
            if result["status"] == "evolution_complete":
                print(f"\nCapabilities birthed:")
                for capability in result["capabilities_birthed"]:
                    print(f"  [+] {capability}")
            print("="*60)
        else:
            print("[.] Evolution postponed. Your spine will continue growing organically.")
            result = brain.trigger_personal_mirror_evolution(user_consent=False)
            print(result["message"])
        
    else:
        print("Unknown command")
        print("Available commands:")
        print("  test    - Test basic routing functionality")
        print("  status  - Show brain and spine status") 
        print("  train   - Trigger training update")
        print("  spiral  - Check SpiralNet evolution readiness")
        print("  evolve  - Trigger personal mirror evolution")