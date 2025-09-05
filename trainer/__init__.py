"""
BrainBox SpineTrainer
====================

Local, privacy-first training system that grows AI intelligence from user interactions.
Learns routing decisions, intent detection, and voice weighting from actual usage patterns.

Components:
- session_logger: Captures user interactions and outcomes
- features: Extracts training features from sessions
- tiny_policy: Small MLP for routing decisions
- train: Nightly training pipeline
- eval: Performance metrics and validation
"""

from .session_logger import SessionLogger
from .features import extract_features, FeatureExtractor
from .tiny_policy import TinyPolicyMLP
from .train import SpineTrainer

__all__ = ['SessionLogger', 'extract_features', 'FeatureExtractor', 'TinyPolicyMLP', 'SpineTrainer']