#!/usr/bin/env python3
"""
Feature Extraction for SpineTrainer
===================================

Converts user interactions into numerical features for training.
Extracts patterns from text, context, and spine state for learning.
"""

import re
import math
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict

class FeatureExtractor:
    """
    Converts session data into training features
    Learns patterns from user behavior and context
    """
    
    def __init__(self):
        # Text pattern matchers
        self.creative_pattern = re.compile(r'\b(creative|brainstorm|ideas?|concept|improvise|invent|design|artistic|visual)\b', re.I)
        self.support_pattern = re.compile(r'\b(overwhelm(ed)?|stuck|support|encourage|calm|soothe|comfort|help|guidance)\b', re.I)
        self.cloud_pattern = re.compile(r'\b(gpt-?4?|claude|openai|cloud|external|advanced|complex|sophisticated)\b', re.I)
        self.search_pattern = re.compile(r'\b(search|find|look\s+up|locate|retrieve|memory|remember)\b', re.I)
        self.planning_pattern = re.compile(r'\b(plan|schedule|organize|roadmap|strategy|structure|outline)\b', re.I)
        
        # Word patterns for intensity
        self.urgent_pattern = re.compile(r'\b(urgent|asap|quickly|fast|immediate|now|emergency)\b', re.I)
        self.casual_pattern = re.compile(r'\b(maybe|perhaps|when\s+you\s+get\s+a\s+chance|no\s+rush)\b', re.I)
        
        # Risk patterns
        self.risky_pattern = re.compile(r'\b(hack|exploit|bypass|override|ignore\s+safety|dangerous)\b', re.I)
        
    def extract_text_features(self, prompt: str) -> np.ndarray:
        """
        Extract text-based features from user prompt
        """
        prompt_lower = prompt.lower()
        prompt_len = len(prompt)
        
        features = [
            # Length features (normalized)
            min(1.0, prompt_len / 500.0),  # Length ratio
            prompt.count('?') / max(1, prompt_len / 50),  # Question density
            prompt.count('!') / max(1, prompt_len / 50),  # Exclamation density
            len([w for w in prompt.split() if w.isupper()]) / max(1, len(prompt.split())),  # Caps ratio
            
            # Intent signals (binary)
            1.0 if self.creative_pattern.search(prompt_lower) else 0.0,
            1.0 if self.support_pattern.search(prompt_lower) else 0.0,
            1.0 if self.cloud_pattern.search(prompt_lower) else 0.0,
            1.0 if self.search_pattern.search(prompt_lower) else 0.0,
            1.0 if self.planning_pattern.search(prompt_lower) else 0.0,
            
            # Intensity signals
            1.0 if self.urgent_pattern.search(prompt_lower) else 0.0,
            1.0 if self.casual_pattern.search(prompt_lower) else 0.0,
            
            # Risk signals
            1.0 if self.risky_pattern.search(prompt_lower) else 0.0,
            
            # Complexity features
            len(prompt.split('.')) / max(1, len(prompt.split())),  # Sentence complexity
            len([w for w in prompt.split() if len(w) > 6]) / max(1, len(prompt.split())),  # Long word ratio
        ]
        
        return np.array(features, dtype=np.float32)
        
    def extract_context_features(self, session: Dict) -> np.ndarray:
        """
        Extract contextual features from session data
        """
        # Time features
        hour = session.get('input', {}).get('time_of_day', 12)
        time_features = [
            math.sin(2 * math.pi * hour / 24),  # Time of day (cyclical)
            math.cos(2 * math.pi * hour / 24),
            1.0 if 9 <= hour <= 17 else 0.0,    # Business hours
            1.0 if hour <= 6 or hour >= 22 else 0.0,  # Late/early hours
        ]
        
        # Session context
        prompt_length = session.get('input', {}).get('prompt_length', 0)
        response_length = session.get('execution', {}).get('response_length', 0)
        latency = session.get('execution', {}).get('latency_ms', 1000)
        
        context_features = [
            min(1.0, prompt_length / 1000.0),    # Normalized prompt length
            min(1.0, response_length / 2000.0),  # Normalized response length  
            min(1.0, latency / 10000.0),         # Normalized latency
        ]
        
        return np.array(time_features + context_features, dtype=np.float32)
        
    def extract_spine_features(self, spine_state: Dict) -> np.ndarray:
        """
        Extract features from current spine state
        """
        if not spine_state:
            return np.zeros(8, dtype=np.float32)
            
        # Active voices count
        active_voices = spine_state.get('active_voices', [])
        voice_count_norm = min(1.0, len(active_voices) / 5.0)
        
        # Voice weights (get weights for known assistants)
        voice_weights = spine_state.get('voice_weights', {})
        search_weight = voice_weights.get('Search Assistant', 1.0) / 2.0  # Normalize to 0.5 baseline
        planning_weight = voice_weights.get('Planning Assistant', 1.0) / 2.0
        creative_weight = voice_weights.get('Creative Assistant', 1.0) / 2.0
        support_weight = voice_weights.get('Support Assistant', 1.0) / 2.0
        
        # User patterns
        patterns = spine_state.get('user_patterns', {})
        session_count = min(1.0, patterns.get('session_count', 0) / 100.0)  # Experience level
        success_rate = patterns.get('success_rate', 0.5)  # Historical success
        
        spine_features = [
            voice_count_norm,
            search_weight,
            planning_weight, 
            creative_weight,
            support_weight,
            session_count,
            success_rate,
            1.0 if len(active_voices) > 2 else 0.0  # Mature spine indicator
        ]
        
        return np.array(spine_features, dtype=np.float32)
        
    def extract_all_features(self, session: Dict) -> np.ndarray:
        """
        Extract complete feature vector for a session
        """
        prompt = session.get('input', {}).get('prompt', '')
        spine_state = session.get('spine_context', {})
        
        text_features = self.extract_text_features(prompt)
        context_features = self.extract_context_features(session)
        spine_features = self.extract_spine_features(spine_state)
        
        # Combine all features
        all_features = np.concatenate([text_features, context_features, spine_features])
        
        return all_features
        
    def create_training_labels(self, session: Dict) -> Dict[str, np.ndarray]:
        """
        Create training labels from session outcomes
        """
        # Intent labels (one-hot encoding)
        intent = session.get('prediction', {}).get('intent', 'planning')
        intent_map = {'search': 0, 'planning': 1, 'creative': 2, 'support': 3, 'other': 4}
        intent_idx = intent_map.get(intent, 4)
        intent_onehot = np.zeros(5)
        intent_onehot[intent_idx] = 1.0
        
        # Intensity (scalar)
        intensity = float(session.get('prediction', {}).get('intensity', 0.5))
        
        # Route decision
        route = session.get('prediction', {}).get('route', 'local_process')
        route_map = {'local_process': 0, 'call_external': 1, 'cloud_consult': 2}
        route_idx = route_map.get(route, 0)
        route_onehot = np.zeros(3)
        route_onehot[route_idx] = 1.0
        
        # Voice distribution (soft targets based on which voices were actually used)
        voices_used = session.get('prediction', {}).get('voices_selected', [])
        voice_distribution = np.zeros(4)  # search, planning, creative, support
        voice_name_map = {
            'Search Assistant': 0,
            'Planning Assistant': 1, 
            'Creative Assistant': 2,
            'Support Assistant': 3
        }
        
        # Normalize based on voices used
        for voice in voices_used:
            if voice in voice_name_map:
                voice_distribution[voice_name_map[voice]] = 1.0
                
        if voice_distribution.sum() > 0:
            voice_distribution = voice_distribution / voice_distribution.sum()
        else:
            voice_distribution[1] = 1.0  # Default to planning
            
        # Risk score (based on outcome success)
        success = session.get('execution', {}).get('success', True)
        risk_score = 0.1 if success else 0.8
        
        return {
            'intent': intent_onehot,
            'intensity': np.array([intensity], dtype=np.float32),
            'route': route_onehot,
            'voice_distribution': voice_distribution.astype(np.float32),
            'risk': np.array([risk_score], dtype=np.float32)
        }

def extract_features(sessions: List[Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Convenience function to extract features from multiple sessions
    
    Returns:
        features: (n_samples, n_features) array
        labels: dict of label arrays for each prediction target
    """
    extractor = FeatureExtractor()
    
    all_features = []
    all_labels = defaultdict(list)
    
    for session in sessions:
        try:
            features = extractor.extract_all_features(session)
            labels = extractor.create_training_labels(session)
            
            all_features.append(features)
            for label_name, label_value in labels.items():
                all_labels[label_name].append(label_value)
                
        except Exception as e:
            print(f"Error processing session: {e}")
            continue
            
    if not all_features:
        return np.array([]), {}
        
    # Convert to numpy arrays
    features_array = np.vstack(all_features)
    labels_dict = {}
    
    for label_name, label_list in all_labels.items():
        if label_list:
            labels_dict[label_name] = np.vstack(label_list)
            
    return features_array, labels_dict

# Testing
if __name__ == "__main__":
    # Test feature extraction
    extractor = FeatureExtractor()
    
    # Mock session data
    test_session = {
        'input': {
            'prompt': 'Help me brainstorm creative ideas for my urgent project!',
            'prompt_length': 52,
            'time_of_day': 14
        },
        'prediction': {
            'intent': 'creative',
            'intensity': 0.8,
            'voices_selected': ['Creative Assistant', 'Planning Assistant'],
            'route': 'local_process'
        },
        'execution': {
            'response_length': 250,
            'latency_ms': 1200,
            'success': True
        },
        'spine_context': {
            'active_voices': ['Search Assistant', 'Planning Assistant', 'Creative Assistant'],
            'voice_weights': {
                'Search Assistant': 1.1,
                'Planning Assistant': 0.9,
                'Creative Assistant': 1.3
            },
            'user_patterns': {
                'session_count': 45,
                'success_rate': 0.82
            }
        }
    }
    
    features = extractor.extract_all_features(test_session)
    labels = extractor.create_training_labels(test_session)
    
    print(f"Features shape: {features.shape}")
    print(f"Features: {features}")
    print(f"Labels: {labels}")
    print(f"Intent: {labels['intent'].argmax()}")  # Should be creative (2)
    print(f"Intensity: {labels['intensity'][0]}")  # Should be 0.8