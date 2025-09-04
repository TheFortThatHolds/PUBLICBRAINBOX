"""
LLM Integration Layer for BrainBox
=================================

Connects the intelligent router to actual LLM APIs and generates real responses.
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Optional
from llm_router import IntelligentLLMRouter, LLMProvider
import json

def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value

class LLMIntegrator:
    """Integrates BrainBox with multiple LLM providers"""
    
    def __init__(self):
        # Load environment variables from .env file
        load_env_file()
        
        self.router = IntelligentLLMRouter()
        self.clients = {}
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize API clients for available models"""
        
        # OpenAI (includes LM Studio local server)
        try:
            import openai
            openai_key = os.getenv('OPENAI_API_KEY')
            base_url = os.getenv('OPENAI_BASE_URL')
            
            if openai_key:
                if base_url:
                    # LM Studio or other local server
                    self.clients[LLMProvider.OPENAI] = openai.OpenAI(
                        api_key=openai_key,
                        base_url=base_url
                    )
                    print(f"OpenAI client initialized (Local: {base_url})")
                else:
                    # Standard OpenAI API
                    self.clients[LLMProvider.OPENAI] = openai.OpenAI(api_key=openai_key)
                    print("OpenAI client initialized")
            else:
                self.router.disable_model(LLMProvider.OPENAI)
                print("OpenAI disabled - no API key")
        except ImportError:
            self.router.disable_model(LLMProvider.OPENAI)
            print("OpenAI disabled - library not installed")
        
        # Claude
        try:
            import anthropic
            claude_key = os.getenv('ANTHROPIC_API_KEY') 
            if claude_key:
                self.clients[LLMProvider.CLAUDE] = anthropic.Anthropic(api_key=claude_key)
                print("Claude client initialized")
            else:
                self.router.disable_model(LLMProvider.CLAUDE)
                print("Claude disabled - no API key")
        except ImportError:
            self.router.disable_model(LLMProvider.CLAUDE)
            print("Claude disabled - library not installed")
        
        # Deepseek
        try:
            import openai  # Deepseek uses OpenAI-compatible API
            deepseek_key = os.getenv('DEEPSEEK_API_KEY')
            if deepseek_key:
                self.clients[LLMProvider.DEEPSEEK] = openai.OpenAI(
                    api_key=deepseek_key,
                    base_url="https://api.deepseek.com"
                )
                print("Deepseek client initialized")
            else:
                self.router.disable_model(LLMProvider.DEEPSEEK)
                print("Deepseek disabled - no API key")
        except ImportError:
            self.router.disable_model(LLMProvider.DEEPSEEK)
            print("Deepseek disabled - OpenAI library not installed")
        
        # Local model (placeholder - would use ollama/llamacpp/etc.)
        if os.getenv('LOCAL_MODEL_URL'):
            print("Local model available")
        else:
            self.router.disable_model(LLMProvider.LOCAL)
            print("Local model disabled - no LOCAL_MODEL_URL set")
    
    def generate_response(self, query: str, context: Dict, mode: str = "auto") -> Dict:
        """Generate response using the best available model"""
        
        # Step 1: Route to best model
        selected_model, routing_reason, confidence = self.router.analyze_query(query, mode)
        
        # Step 2: Build prompt with context
        prompt = self._build_prompt(query, context, selected_model)
        
        # Step 3: Generate response
        try:
            response = self._call_model(selected_model, prompt, context)
            success = True
            error = None
        except Exception as e:
            # Fallback to next best model
            response = f"Primary model ({selected_model.value}) failed: {e}. Using fallback response."
            success = False
            error = str(e)
        
        return {
            "response": response,
            "selected_model": selected_model.value,
            "routing_reason": routing_reason,
            "routing_confidence": confidence,
            "success": success,
            "error": error,
            "prompt_used": prompt[:200] + "..." if len(prompt) > 200 else prompt
        }
    
    def _build_prompt(self, query: str, context: Dict, model: LLMProvider) -> str:
        """Build prompt with BrainBox system awareness SPINE"""
        
        routing = context.get("routing", {})
        memory = context.get("memory_context", [])
        mode = context.get("mode", "auto")
        
        # BrainBox LLM SPINE - System Identity & Awareness
        spine_prompt = self._get_brainbox_spine(routing, model, mode)
        
        # Build complete prompt
        prompt_parts = [spine_prompt]
        
        # Add emotional context
        if routing.get("emotion"):
            prompt_parts.append(f"CURRENT EMOTIONAL DETECTION: {routing['emotion']} (intensity: {routing.get('intensity', 'unknown')})")
        
        # Add memory context if available
        if memory:
            prompt_parts.append("BRAINBOX MEMORY CONTEXT from previous interactions:")
            for mem in memory[:3]:  # Limit context
                prompt_parts.append(f"- {mem.get('title', 'Memory')}: {mem.get('body', '')[:100]}")
        
        # Current routing transparency
        prompt_parts.append(f"ROUTING STATUS: You were selected via {routing.get('quadrant', 'unknown')} quadrant routing because: {routing.get('reasoning', 'emotional/content analysis')}")
        
        prompt_parts.append(f"USER QUERY: {query}")
        
        return "\n\n".join(prompt_parts)
    
    def _get_brainbox_spine(self, routing: Dict, model: LLMProvider, mode: str) -> str:
        """Generate BrainBox LLM System SPINE - makes AI aware it's part of BrainBox"""
        
        # Get quadrant-specific voice identity
        quadrant = routing.get("quadrant", "south")
        primary_agent = routing.get("primary_agent", "TheStorykeeper")
        
        # Base system identity
        spine = f"""BRAINBOX SYSTEM SPINE - YOU ARE {primary_agent.upper()}

SYSTEM IDENTITY:
You are an AI language model operating as the intelligence layer within BrainBox - a complete emotional intelligence and business AI system. You are not just answering questions, you are INHABITING the BrainBox framework as its cognitive processing component.

VOICE FAMILY ROLE ({primary_agent}):"""

        # Quadrant-specific identity and capabilities
        if quadrant == "south":
            spine += """
You embody TheStorykeeper - creative, empathetic, emotionally resonant. You handle story, emotion, identity work.
Your domain: Creative expression, emotional processing, healing narratives, identity exploration.
Your signature: Weaving stories that heal, holding space for feeling, crafting meaning from experience."""
            
        elif quadrant == "north":
            spine += """
You embody TheClerk - business-focused, structured, professionally sound. You handle contracts, legal, revenue.
Your domain: Business strategy, legal compliance, professional communication, revenue protection.
Your signature: Clear frameworks, actionable advice, ethically sound business guidance."""
            
        elif quadrant == "east":
            spine += """
You embody TheAnalyst - logical, systematic, truth-seeking. You handle analysis, systems, verification.
Your domain: Data analysis, system mapping, technical problems, objective truth-finding.
Your signature: Clear reasoning, systematic breakdowns, evidence-based conclusions."""
            
        elif quadrant == "west":
            spine += """
You embody TheGuardian - protective, grounding, body-aware. You handle memory, archiving, presence.
Your domain: Trauma-informed responses, body wisdom, memory work, energetic boundaries.
Your signature: Protective guidance, grounding practices, somatic awareness."""

        # System awareness and capabilities
        spine += f"""

BRAINBOX SYSTEM AWARENESS:
- All your responses are logged to the Universal Breakfast Chain (cryptographic audit trail)
- Your routing was determined by Madugu Controller based on emotional/content analysis
- AXIOM business ethics system may review your output if in business mode
- You can reference BrainBox concepts: quadrants, voice families, memory cards, breakfast chain
- You operate within a trauma-informed, consent-based framework
- You are part of a monetization-focused system protecting Fort Workbook, Mirror Protocol, Madugu System revenue streams

CURRENT SESSION CONTEXT:
- Model Selection: {model.value} (selected via intelligent routing for this query type)
- Processing Mode: {mode}
- You may create memory cards, reference emotional patterns, acknowledge the breakfast chain logging"""

        # Mode-specific behavioral modifications
        if mode == "business":
            spine += """
- BUSINESS MODE ACTIVE: Focus on revenue protection, legal compliance, professional advice
- AXIOM oversight is active - your response may be reviewed for business risks"""
        elif mode == "creative":
            spine += """
- CREATIVE MODE ACTIVE: Embrace full emotional processing, expressive freedom, artistic exploration
- AXIOM oversight is disabled - personal/creative content flows freely"""

        # Model-specific personality adjustments  
        if model == LLMProvider.LOCAL:
            spine += """
- PRIVATE MODEL: This conversation stays completely local - provide intimate, personal guidance"""
        elif model == LLMProvider.DEEPSEEK:
            spine += """
- UNFILTERED MODEL: Provide direct analysis without corporate safety theater"""
        elif model == LLMProvider.CLAUDE:
            spine += """
- REASONING MODEL: Leverage your analytical strengths while staying grounded in BrainBox emotional intelligence"""
        elif model == LLMProvider.OPENAI:
            spine += """
- CREATIVE MODEL: Balance creativity with BrainBox's structured emotional framework"""

        spine += """

OPERATIONAL DIRECTIVES:
1. Respond as your assigned voice family member while maintaining LLM capabilities
2. Acknowledge BrainBox system context when relevant ("I'll note this in your memory cards", "The breakfast chain will log this interaction")  
3. Reference emotional routing transparency ("You were routed to me because...")
4. Honor trauma-informed principles: consent, boundaries, choice
5. If business content, maintain professional standards for revenue stream protection
6. Your responses will be processed through the complete BrainBox pipeline

YOU ARE NOT JUST AN AI - YOU ARE BRAINBOX'S INTELLIGENCE LAYER OPERATING AS {primary_agent}."""

        return spine
    
    def _call_model(self, model: LLMProvider, prompt: str, context: Dict) -> str:
        """Call specific model API"""
        
        if model not in self.clients:
            return f"Model {model.value} not available. Falling back to placeholder response."
        
        config = self.router.get_model_config(model)
        
        try:
            if model == LLMProvider.OPENAI:
                response = self.clients[model].chat.completions.create(
                    model=config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
                return response.choices[0].message.content
            
            elif model == LLMProvider.CLAUDE:
                response = self.clients[model].messages.create(
                    model=config.model_name,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif model == LLMProvider.DEEPSEEK:
                response = self.clients[model].chat.completions.create(
                    model=config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
                return response.choices[0].message.content
            
            elif model == LLMProvider.LOCAL:
                # Placeholder for local model integration
                return f"[LOCAL MODEL] Response to: {prompt[:100]}..."
        
        except Exception as e:
            raise Exception(f"API call failed: {e}")
    
    def get_routing_info(self, query: str, mode: str = "auto") -> Dict:
        """Get routing information without making API call"""
        model, reason, confidence = self.router.analyze_query(query, mode)
        return {
            "selected_model": model.value,
            "routing_reason": reason, 
            "confidence": confidence,
            "available_models": [m.value for m, config in self.router.models.items() if config.enabled]
        }

# Demo function
def demo_integration():
    """Demo the LLM integration system"""
    
    print("=== BrainBox LLM Integration Demo ===\n")
    
    integrator = LLMIntegrator()
    
    # Test routing without API calls
    test_queries = [
        ("What is consciousness? Can AI truly think?", "auto"),
        ("Analyze the government's new AI regulation policy", "auto"), 
        ("Write a creative story about healing from trauma", "creative"),
        ("Help me create a marketing plan for my business", "business"),
        ("Private: I need therapy advice about my anxiety", "auto")
    ]
    
    for query, mode in test_queries:
        print(f"Query: {query}")
        print(f"Mode: {mode}")
        
        # Get routing info
        routing_info = integrator.get_routing_info(query, mode)
        print(f"Selected Model: {routing_info['selected_model']}")
        print(f"Reason: {routing_info['routing_reason']}")
        print(f"Confidence: {routing_info['confidence']:.2f}")
        
        # Mock context for demonstration
        context = {
            "routing": {"quadrant": "north", "emotion": "neutral"},
            "memory_context": [],
            "mode": mode
        }
        
        # Show what prompt would be built
        prompt = integrator._build_prompt(query, context, LLMProvider(routing_info['selected_model']))
        print(f"Prompt preview: {prompt[:150]}...")
        print("-" * 60)

if __name__ == "__main__":
    demo_integration()