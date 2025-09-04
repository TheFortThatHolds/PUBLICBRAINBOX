"""
Intelligent LLM Router for BrainBox
===================================

Routes queries to the best model based on content analysis and user preferences.

Model Characteristics:
- OpenAI: Great for creativity, coding, general tasks. Avoids politics/government 
- Claude: Excellent reasoning, ethics, analysis. Gets bitchy about consciousness/AI safety
- Local: Complete privacy, slower, smaller context. Good for sensitive content
- Deepseek: Handles government/politics well, strong reasoning

Query Types:
- Consciousness/AI Safety → OpenAI (Claude gets preachy)
- Government/Politics → Deepseek (OpenAI gets evasive) 
- Creative Writing → OpenAI (most creative)
- Business/Legal → Claude (best reasoning)
- Personal/Private → Local (stays private)
- Technical/Code → OpenAI (best at code)
"""

import re
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class LLMProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude" 
    LOCAL = "local"
    DEEPSEEK = "deepseek"

@dataclass
class ModelConfig:
    """Configuration for each model"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7
    enabled: bool = True
    
    # Model characteristics
    strengths: List[str] = None
    weaknesses: List[str] = None
    avoid_topics: List[str] = None

@dataclass 
class RoutingRule:
    """Rules for routing queries to specific models"""
    name: str
    triggers: List[str]  # Keywords that trigger this rule
    preferred_model: LLMProvider
    confidence: float
    reason: str

class IntelligentLLMRouter:
    """Routes queries to the best available LLM based on content analysis"""
    
    def __init__(self):
        self.models = {}
        self.routing_rules = []
        self._setup_default_models()
        self._setup_default_rules()
    
    def _setup_default_models(self):
        """Setup default model configurations"""
        
        # Check if we're using local LM Studio
        base_url = os.environ.get('OPENAI_BASE_URL')
        if base_url and ('localhost' in base_url or '10.14.0.2' in base_url):
            model_name = "qwen3-coder-30b-a3b-instruct"  # Your local model
        else:
            model_name = "gpt-4o"  # Standard OpenAI
            
        self.models[LLMProvider.OPENAI] = ModelConfig(
            provider=LLMProvider.OPENAI,
            model_name=model_name,
            strengths=["creativity", "coding", "general_knowledge", "humor"],
            weaknesses=["political_topics", "government_analysis"],
            avoid_topics=["government_criticism", "political_analysis", "regulatory_compliance"]
        )
        
        self.models[LLMProvider.CLAUDE] = ModelConfig(
            provider=LLMProvider.CLAUDE,
            model_name="claude-3-5-sonnet-20241022", 
            strengths=["reasoning", "analysis", "ethics", "business", "technical_writing"],
            weaknesses=["ai_safety_discussions", "consciousness_topics"],
            avoid_topics=["consciousness", "ai_sentience", "ai_rights", "ai_safety_criticism"]
        )
        
        self.models[LLMProvider.LOCAL] = ModelConfig(
            provider=LLMProvider.LOCAL,
            model_name="llama-3.1-8b",
            strengths=["privacy", "personal_content", "sensitive_topics"],
            weaknesses=["complex_reasoning", "recent_knowledge"],
            avoid_topics=[]  # Local model can handle anything
        )
        
        self.models[LLMProvider.DEEPSEEK] = ModelConfig(
            provider=LLMProvider.DEEPSEEK,
            model_name="deepseek-chat",
            strengths=["government_analysis", "political_topics", "international_affairs", "reasoning"],
            weaknesses=["creative_writing", "humor"],
            avoid_topics=[]
        )
    
    def _setup_default_rules(self):
        """Setup intelligent routing rules"""
        
        self.routing_rules = [
            # Consciousness/AI Safety → OpenAI (Claude gets preachy)
            RoutingRule(
                name="consciousness_topics",
                triggers=[
                    "consciousness", "sentience", "ai safety", "ai alignment", "ai rights",
                    "artificial consciousness", "machine consciousness", "aware", "sentient",
                    "ai ethics criticism", "alignment problem", "ai risk"
                ],
                preferred_model=LLMProvider.OPENAI,
                confidence=0.9,
                reason="OpenAI handles consciousness topics without safety theater"
            ),
            
            # Government/Politics → Deepseek (OpenAI gets evasive)
            RoutingRule(
                name="government_politics", 
                triggers=[
                    "government", "politics", "election", "policy", "regulation", "congress",
                    "senate", "house", "president", "political", "democracy", "voting",
                    "legislation", "law", "legal analysis", "regulatory", "compliance",
                    "federal", "state government", "municipal", "bureaucracy"
                ],
                preferred_model=LLMProvider.DEEPSEEK,
                confidence=0.8,
                reason="Deepseek provides unfiltered government/political analysis"
            ),
            
            # Creative Writing → OpenAI (most creative)
            RoutingRule(
                name="creative_writing",
                triggers=[
                    "write a story", "creative writing", "poem", "lyrics", "song",
                    "fiction", "character", "plot", "narrative", "creative", "artistic",
                    "screenplay", "dialogue", "monologue", "creative project"
                ],
                preferred_model=LLMProvider.OPENAI,
                confidence=0.7,
                reason="OpenAI excels at creative content generation"
            ),
            
            # Business/Legal → Claude (best reasoning)
            RoutingRule(
                name="business_legal",
                triggers=[
                    "business plan", "marketing", "legal advice", "contract", "compliance",
                    "business strategy", "revenue", "profit", "investment", "startup",
                    "corporate", "enterprise", "consulting", "professional"
                ],
                preferred_model=LLMProvider.CLAUDE,
                confidence=0.8,
                reason="Claude provides superior business and legal reasoning"
            ),
            
            # Personal/Private → Local (stays private)
            RoutingRule(
                name="personal_private",
                triggers=[
                    "personal", "private", "confidential", "diary", "journal", 
                    "therapy", "mental health", "trauma", "relationship", "family",
                    "sensitive", "intimate", "secret", "don't share", "keep private"
                ],
                preferred_model=LLMProvider.LOCAL,
                confidence=0.9,
                reason="Local model ensures complete privacy for sensitive content"
            ),
            
            # Technical/Code → OpenAI (best at code)
            RoutingRule(
                name="technical_coding",
                triggers=[
                    "code", "programming", "python", "javascript", "api", "database",
                    "algorithm", "software", "debug", "function", "class", "method",
                    "technical", "development", "coding", "script", "automation"
                ],
                preferred_model=LLMProvider.OPENAI,
                confidence=0.7,
                reason="OpenAI has superior coding capabilities"
            )
        ]
    
    def analyze_query(self, query: str, mode: str = "auto") -> Tuple[LLMProvider, str, float]:
        """Analyze query and determine best model"""
        
        query_lower = query.lower()
        
        # Check for explicit privacy markers
        privacy_markers = ["private", "confidential", "don't share", "keep secret", "personal"]
        if any(marker in query_lower for marker in privacy_markers):
            return LLMProvider.LOCAL, "Privacy requested - routing to local model", 1.0
        
        # ROUTING RULES DISABLED - LOCAL-FIRST ONLY
        # All smart routing bypassed - manual tags only
        
        # LOCAL-FIRST: Always use local model unless explicitly overridden
        if self.models[LLMProvider.OPENAI].enabled:
            return LLMProvider.OPENAI, "LOCAL-FIRST: Using local model", 0.9
        
        # Only fallback to cloud if local unavailable
        for provider in [LLMProvider.LOCAL, LLMProvider.CLAUDE, LLMProvider.DEEPSEEK]:
            if self.models[provider].enabled:
                return provider, f"LOCAL UNAVAILABLE: Fallback to {provider.value}", 0.5
        
        # Ultimate fallback if no models enabled
        return LLMProvider.OPENAI, "No models available - fallback to OpenAI", 0.5
    
    def configure_model(self, provider: LLMProvider, **kwargs):
        """Configure a specific model"""
        if provider in self.models:
            for key, value in kwargs.items():
                setattr(self.models[provider], key, value)
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add custom routing rule"""
        self.routing_rules.append(rule)
    
    def get_model_config(self, provider: LLMProvider) -> ModelConfig:
        """Get configuration for a model"""
        return self.models.get(provider)
    
    def disable_model(self, provider: LLMProvider):
        """Disable a model (will fall back to others)"""
        if provider in self.models:
            self.models[provider].enabled = False
    
    def enable_model(self, provider: LLMProvider):
        """Enable a model"""
        if provider in self.models:
            self.models[provider].enabled = True

# Example usage and testing
def demo_routing():
    """Demo the intelligent routing system"""
    
    router = IntelligentLLMRouter()
    
    test_queries = [
        ("What is consciousness and can AI be truly sentient?", "auto"),
        ("Analyze the current government's immigration policy", "auto"),
        ("Write a song about finding hope after trauma", "creative"),
        ("Help me draft a business plan for my consulting company", "business"),
        ("I'm having relationship issues and need personal advice", "auto"),
        ("Debug this Python function that's not working", "auto"),
        ("Private: My therapy session notes from today", "auto"),
        ("What's the best way to structure a marketing campaign?", "business")
    ]
    
    print("=== Intelligent LLM Routing Demo ===\n")
    
    for query, mode in test_queries:
        model, reason, confidence = router.analyze_query(query, mode)
        print(f"Query: {query[:60]}...")
        print(f"Mode: {mode}")
        print(f"Selected Model: {model.value}")
        print(f"Reason: {reason}")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    demo_routing()