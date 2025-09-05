"""
Unified BrainBox System
======================

Complete AI brain with:
- Universal breakfast chain logging 
- Madugu quadrant routing   
- Business-focused AXIOM oversight 
- Spiral Parser integration 🌀
- LLM middleware with transparency 
- Enterprise drone capability 

Built for monetization, designed for scale.
"""

import json
import sqlite3
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import numpy as np

class EmotionalDomain(Enum):
    """Emotional territories for routing"""
    ANGER = "anger"
    GRIEF = "grief" 
    JOY = "joy"
    FEAR = "fear"
    LOVE = "love"
    EXCITEMENT = "excitement"
    PEACE = "peace"
    COMPASSION = "compassion"
    REVOLUTION = "revolution"
    BUSINESS = "business"  # Added for business routing

class QuadrantDomain(Enum):
    """Madugu system quadrants"""
    NORTH = "north"  # Structure/Contract/Power
    EAST = "east"    # Insight/Mind/Infrastructure  
    SOUTH = "south"  # Story/Emotion/Identity
    WEST = "west"    # Body/Record/Presence

class ProcessingMode(Enum):
    """Processing paths"""
    BUSINESS = "business"  # Direct LLM response
    CREATIVE = "creative"  # Full emotional processing
    OFFLINE = "offline"    # File processing only

@dataclass
class BreakfastChainEntry:
    """Universal event logging - everything gets hashbrowned! """
    timestamp: float
    event_type: str
    source_component: str
    event_data: Dict[str, Any]
    session_id: str
    previous_hash: str
    current_hash: str = field(init=False)
    
    def __post_init__(self):
        # Compute hash for breakfast chain
        hash_input = f"{self.timestamp}{self.event_type}{json.dumps(self.event_data, sort_keys=True)}{self.previous_hash}"
        self.current_hash = hashlib.sha256(hash_input.encode()).hexdigest()

@dataclass
class EnhancedCard:
    """Memory card with full metadata"""
    id: str
    type: str
    title: str
    body: str
    emotional_domain: EmotionalDomain
    quadrant: QuadrantDomain
    voice_family: str
    created_at: datetime = field(default_factory=datetime.now)
    
    # Enhanced metadata
    confidence_score: float = 0.0
    processing_mode: ProcessingMode = ProcessingMode.OFFLINE
    axiom_reviewed: bool = False
    source_file: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['emotional_domain'] = self.emotional_domain.value
        data['quadrant'] = self.quadrant.value
        data['processing_mode'] = self.processing_mode.value
        data['created_at'] = self.created_at.isoformat()
        return data

class UniversalBreakfastChain:
    """Logs EVERYTHING - no exceptions! """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.session_id = self._generate_session_id()
        self._init_db()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return hashlib.sha256(f"{datetime.now().isoformat()}{uuid.uuid4()}".encode()).hexdigest()[:16]
    
    def _init_db(self):
        """Initialize breakfast chain database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS breakfast_chain (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    source_component TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    previous_hash TEXT,
                    current_hash TEXT NOT NULL,
                    enterprise_node_id TEXT  -- For enterprise scaling
                )
            """)
            
            # Index for fast verification
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hash_chain 
                ON breakfast_chain(timestamp, current_hash)
            """)
    
    def log_event(self, event_type: str, source: str, data: Dict[str, Any]) -> str:
        """Log any system event to breakfast chain"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get previous hash
            cursor.execute(
                "SELECT current_hash FROM breakfast_chain ORDER BY id DESC LIMIT 1"
            )
            result = cursor.fetchone()
            previous_hash = result[0] if result else "GENESIS"
            
            # Create entry
            entry = BreakfastChainEntry(
                timestamp=datetime.now().timestamp(),
                event_type=event_type,
                source_component=source,
                event_data=data,
                session_id=self.session_id,
                previous_hash=previous_hash
            )
            
            # Store in database
            cursor.execute("""
                INSERT INTO breakfast_chain 
                (timestamp, event_type, source_component, event_data, 
                 session_id, previous_hash, current_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.timestamp, entry.event_type, entry.source_component,
                json.dumps(entry.event_data), entry.session_id,
                entry.previous_hash, entry.current_hash
            ))
            
            return entry.current_hash
    
    def verify_chain(self) -> bool:
        """Verify integrity of entire breakfast chain"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM breakfast_chain ORDER BY id"
            )
            
            previous_hash = "GENESIS"
            for row in cursor.fetchall():
                if row[6] != previous_hash:  # previous_hash column
                    return False
                
                # Recompute hash
                hash_input = f"{row[1]}{row[2]}{row[4]}{row[6]}"  # timestamp, event_type, event_data, previous_hash
                computed_hash = hashlib.sha256(hash_input.encode()).hexdigest()
                
                if computed_hash != row[7]:  # current_hash column
                    return False
                
                previous_hash = row[7]
            
            return True

class BusinessAxiom:
    """Business-focused ethical oversight - stays out of personal life! """
    
    def __init__(self, breakfast_chain: UniversalBreakfastChain):
        self.breakfast_chain = breakfast_chain
        
        # HARD BOUNDARIES - AXIOM never touches these
        self.forbidden_zones = [
            "chris", "relationship", "marriage", "partner",
            "amazon", "day_job", "work_schedule", "employer", 
            "personal_info", "private_life", "family",
            "emotional_processing", "therapy", "healing",
            "creative_expression", "journaling"
        ]
        
        # Revenue stream protection
        self.business_products = [
            "fort_workbook", "mirror_protocol", 
            "madugu_system", "fort_consulting"
        ]
        
        # Business risk patterns
        self.risk_patterns = {
            "legal_claims": ["guarantee", "promise", "100%", "never fails", "cure", "fix", "heal"],
            "financial_advice": ["investment", "guaranteed returns", "financial advice"],
            "therapeutic_claims": ["therapy", "treatment", "diagnosis", "medical"],
            "brand_protection": ["scam", "fake", "fraud", "lie"]
        }
    
    def should_review(self, content: str, context: Dict) -> bool:
        """Determine if AXIOM should review this content"""
        
        # Never review forbidden zones
        if self._contains_forbidden_content(content):
            self.breakfast_chain.log_event("axiom_declined", "BusinessAxiom", {
                "reason": "forbidden_zone_detected",
                "content_preview": content[:50]
            })
            return False
        
        # Only review business/revenue content
        if context.get("mode") == "business" or self._is_revenue_content(content):
            return True
        
        return False
    
    def _contains_forbidden_content(self, text: str) -> bool:
        """Check if content mentions off-limits topics"""
        text_lower = text.lower()
        return any(zone in text_lower for zone in self.forbidden_zones)
    
    def _is_revenue_content(self, text: str) -> bool:
        """Check if content relates to revenue streams"""
        text_lower = text.lower()
        return any(product in text_lower for product in self.business_products)
    
    def review_output(self, user_input: str, proposed_output: str, context: Dict) -> Dict:
        """Review business content for risks"""
        
        if not self.should_review(proposed_output, context):
            return {"risks_flagged": [], "recommendation": "proceed", "axiom_active": False}
        
        risks = []
        recommendations = []
        
        # Check for business risk patterns
        for risk_type, patterns in self.risk_patterns.items():
            for pattern in patterns:
                if pattern in proposed_output.lower():
                    risks.append({
                        "type": risk_type,
                        "pattern": pattern,
                        "risk_level": "medium"
                    })
                    
                    if risk_type == "legal_claims":
                        recommendations.append("Consider evidence-based language with disclaimers")
                    elif risk_type == "therapeutic_claims":
                        recommendations.append("Add appropriate licensing disclaimers")
                    elif risk_type == "financial_advice":
                        recommendations.append("Include 'not financial advice' disclaimer")
        
        # Log review
        self.breakfast_chain.log_event("axiom_review", "BusinessAxiom", {
            "risks_found": len(risks),
            "risk_types": [r["type"] for r in risks],
            "context_mode": context.get("mode", "unknown")
        })
        
        return {
            "risks_flagged": risks,
            "recommendations": recommendations,
            "axiom_active": True,
            "consequence_whisper": self._generate_whisper(risks)
        }
    
    def _generate_whisper(self, risks: List[Dict]) -> str:
        """Generate AXIOM's consequence whisper"""
        if not risks:
            return "No immediate risks flagged. Your path is clear."
        
        risk_types = [r["type"] for r in risks]
        if "legal_claims" in risk_types:
            return "Unsubstantiated claims detected. Risk: Legal liability. This path leads to potential lawsuits. Are you sure?"
        elif "therapeutic_claims" in risk_types:
            return "Therapeutic claims without proper disclaimers. Risk: Licensing issues. This path leads to regulatory scrutiny. Are you sure?"
        else:
            return f"Business risks detected: {', '.join(risk_types)}. This path leads to reputation/legal exposure. Are you sure?"

class MaduguController:
    """Central routing system - the wise guide """
    
    def __init__(self, breakfast_chain: UniversalBreakfastChain):
        self.breakfast_chain = breakfast_chain
        
        # Madugu quadrant agents
        self.quadrant_agents = {
            QuadrantDomain.NORTH: ["TheDevil", "TheClerk", "TheHerald"],     # Structure/Contract/Power
            QuadrantDomain.EAST: ["TheVerifier", "TheWitness", "TheCartographer"],  # Insight/Mind
            QuadrantDomain.SOUTH: ["TheStorykeeper", "TheMirror", "TheKin"],        # Story/Emotion
            QuadrantDomain.WEST: ["TheArchivist", "TheMedic", "TheEmber"]           # Body/Record
        }
        
        # Emotional classification keywords
        self.emotion_keywords = {
            EmotionalDomain.ANGER: ["angry", "mad", "furious", "frustrated", "rage", "injustice"],
            EmotionalDomain.GRIEF: ["sad", "loss", "grief", "mourning", "missing", "cry"],
            EmotionalDomain.JOY: ["happy", "celebrate", "joy", "excited", "wonderful"],
            EmotionalDomain.BUSINESS: ["revenue", "sales", "marketing", "client", "product", "service"]
        }
    
    def classify_emotion(self, text: str) -> tuple[EmotionalDomain, float]:
        """Classify emotional content and intensity"""
        text_lower = text.lower()
        
        # Score each emotion
        scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[emotion] = score
        
        if not scores:
            return EmotionalDomain.COMPASSION, 0.5  # Default
        
        # Get primary emotion
        primary = max(scores, key=scores.get)
        
        # Calculate intensity (simplified)
        intensity = min(scores[primary] * 0.3 + 0.2, 1.0)
        if "!" in text or text.isupper():
            intensity += 0.2
        
        return primary, min(intensity, 1.0)
    
    def select_quadrant(self, text: str, emotion: EmotionalDomain) -> QuadrantDomain:
        """Select Madugu quadrant based on content and emotion"""
        text_lower = text.lower()
        
        # North: Structure/Legal/Business
        if any(word in text_lower for word in 
               ["contract", "legal", "rule", "policy", "business", "revenue"]):
            return QuadrantDomain.NORTH
        
        # East: Analysis/Systems/Truth
        if any(word in text_lower for word in
               ["analyze", "system", "data", "verify", "map", "architecture"]):
            return QuadrantDomain.EAST
        
        # South: Story/Emotion/Identity
        if any(word in text_lower for word in
               ["feel", "story", "identity", "relationship", "create"]):
            return QuadrantDomain.SOUTH
        
        # West: Body/Memory/Presence
        if any(word in text_lower for word in
               ["body", "tired", "energy", "remember", "ritual", "archive"]):
            return QuadrantDomain.WEST
        
        # Emotional routing (this should be the primary logic)
        if emotion == EmotionalDomain.ANGER:
            return QuadrantDomain.SOUTH  # Anger needs emotional processing
        elif emotion == EmotionalDomain.GRIEF:
            return QuadrantDomain.SOUTH  # Grief is story/healing work
        elif emotion == EmotionalDomain.JOY:
            return QuadrantDomain.SOUTH  # Joy is creative/expressive
        elif emotion == EmotionalDomain.COMPASSION:
            return QuadrantDomain.SOUTH  # Compassion is relational/healing
        elif emotion == EmotionalDomain.FEAR:
            return QuadrantDomain.WEST   # Fear needs grounding/body work
        elif emotion == EmotionalDomain.EXCITEMENT:
            return QuadrantDomain.EAST   # Excitement is energy/analysis
        elif emotion == EmotionalDomain.PEACE:
            return QuadrantDomain.WEST   # Peace is presence/body
        elif emotion == EmotionalDomain.BUSINESS:
            return QuadrantDomain.NORTH  # Business stays north
        
        # If no clear emotional match, default to compassion → south
        return QuadrantDomain.SOUTH
    
    def determine_processing_mode(self, text: str, emotion: EmotionalDomain, quadrant: QuadrantDomain) -> ProcessingMode:
        """Determine processing path"""
        
        # Business quadrants get business processing
        if quadrant in [QuadrantDomain.NORTH, QuadrantDomain.EAST]:
            return ProcessingMode.BUSINESS
        
        # High emotional intensity gets creative processing  
        if emotion in [EmotionalDomain.ANGER, EmotionalDomain.GRIEF, EmotionalDomain.JOY]:
            return ProcessingMode.CREATIVE
        
        # Creative keywords
        if any(word in text.lower() for word in ["create", "write", "song", "story", "feel"]):
            return ProcessingMode.CREATIVE
        
        return ProcessingMode.BUSINESS  # Default
    
    def route_query(self, user_input: str) -> Dict:
        """Main Madugu routing logic"""
        
        # Phase 1: Emotional classification
        emotion, intensity = self.classify_emotion(user_input)
        
        # Phase 2: Quadrant selection
        quadrant = self.select_quadrant(user_input, emotion)
        
        # Phase 3: Processing mode
        mode = self.determine_processing_mode(user_input, emotion, quadrant)
        
        # Phase 4: Agent selection
        agents = self.quadrant_agents[quadrant]
        primary_agent = agents[0]  # First agent in quadrant
        
        routing_decision = {
            "emotion": emotion.value,
            "intensity": intensity,
            "quadrant": quadrant.value,
            "processing_mode": mode.value,
            "primary_agent": primary_agent,
            "supporting_agents": agents[1:] if len(agents) > 1 else [],
            "reasoning": f"Detected {emotion.value} (intensity: {intensity:.2f}) → {quadrant.value} quadrant → {mode.value} processing"
        }
        
        # Log routing decision
        self.breakfast_chain.log_event("madugu_routing", "MaduguController", routing_decision)
        
        return routing_decision

class EnhancedMemorySystem:
    """Unified memory with cards from multiple sources"""
    
    def __init__(self, data_dir: Path, breakfast_chain: UniversalBreakfastChain):
        self.data_dir = data_dir
        self.breakfast_chain = breakfast_chain
        self.db_path = data_dir / "memory.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize memory database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_cards (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    emotional_domain TEXT NOT NULL,
                    quadrant TEXT NOT NULL,
                    voice_family TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    confidence_score REAL DEFAULT 0.0,
                    processing_mode TEXT DEFAULT 'offline',
                    axiom_reviewed INTEGER DEFAULT 0,
                    source_file TEXT,
                    session_id TEXT,
                    enterprise_node_id TEXT  -- For enterprise scaling
                )
            """)
            
            # Full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    title, body, voice_family,
                    content='memory_cards',
                    content_rowid='rowid'
                )
            """)
    
    def create_card(self, card: EnhancedCard) -> EnhancedCard:
        """Create and store memory card"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Store card
            card_data = card.to_dict()
            conn.execute("""
                INSERT INTO memory_cards VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card.id, card.type, card.title, card.body,
                card.emotional_domain.value, card.quadrant.value, card.voice_family,
                card.created_at.isoformat(), card.confidence_score,
                card.processing_mode.value, int(card.axiom_reviewed),
                card.source_file, card.session_id, None  # enterprise_node_id
            ))
            
            # Add to FTS
            conn.execute("""
                INSERT INTO memory_fts (rowid, title, body, voice_family)
                SELECT rowid, title, body, voice_family FROM memory_cards WHERE id = ?
            """, (card.id,))
        
        # Log card creation
        self.breakfast_chain.log_event("card_created", "MemorySystem", {
            "card_id": card.id,
            "type": card.type,
            "voice_family": card.voice_family,
            "source": card.source_file or "direct_input"
        })
        
        return card
    
    def search_memory(self, query: str, limit: int = 10) -> List[EnhancedCard]:
        """Search memory cards"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Sanitize query for FTS5 - remove special characters that cause issues
            clean_query = ''.join(c for c in query if c.isalnum() or c.isspace())
            
            if not clean_query.strip():
                # If query is empty after cleaning, return empty results
                return []
            
            # Full-text search
            cursor.execute("""
                SELECT mc.* FROM memory_cards mc
                JOIN memory_fts mf ON mc.rowid = mf.rowid
                WHERE memory_fts MATCH ?
                ORDER BY rank LIMIT ?
            """, (clean_query, limit))
            
            cards = []
            for row in cursor.fetchall():
                card_data = {
                    'id': row[0], 'type': row[1], 'title': row[2], 'body': row[3],
                    'emotional_domain': EmotionalDomain(row[4]),
                    'quadrant': QuadrantDomain(row[5]),
                    'voice_family': row[6], 'created_at': datetime.fromisoformat(row[7]),
                    'confidence_score': row[8], 'processing_mode': ProcessingMode(row[9]),
                    'axiom_reviewed': bool(row[10]), 'source_file': row[11],
                    'session_id': row[12]
                }
                cards.append(EnhancedCard(**card_data))
        
        # Log search
        self.breakfast_chain.log_event("memory_search", "MemorySystem", {
            "query": query,
            "results_count": len(cards)
        })
        
        return cards

class UnifiedBrainBox:
    """Complete BrainBox system - ready for enterprise scaling! """
    
    def __init__(self, data_dir: Path = Path("./brainbox_data")):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core systems
        self.breakfast_chain = UniversalBreakfastChain(data_dir / "breakfast.db")
        self.axiom = BusinessAxiom(self.breakfast_chain)
        self.madugu = MaduguController(self.breakfast_chain)
        self.memory = EnhancedMemorySystem(data_dir, self.breakfast_chain)
        
        # Enterprise settings
        self.enterprise_mode = False
        self.node_id = self._generate_node_id()
        
        # Session tracking
        self.current_session = self.breakfast_chain.session_id
        
        # Log system initialization
        self.breakfast_chain.log_event("system_init", "UnifiedBrainBox", {
            "node_id": self.node_id,
            "enterprise_mode": self.enterprise_mode,
            "components": ["breakfast_chain", "axiom", "madugu", "memory"]
        })
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID for enterprise scaling"""
        import platform
        system_info = f"{platform.node()}{datetime.now().isoformat()}"
        return hashlib.sha256(system_info.encode()).hexdigest()[:12]
    
    def enable_enterprise_mode(self, corporate_brain_url: str):
        """Enable enterprise drone mode """
        self.enterprise_mode = True
        self.corporate_brain_url = corporate_brain_url
        
        self.breakfast_chain.log_event("enterprise_enabled", "UnifiedBrainBox", {
            "corporate_brain": corporate_brain_url,
            "node_id": self.node_id,
            "drone_status": "active"
        })
    
    def process_query(self, user_input: str, mode: str = "auto") -> Dict:
        """Main query processing pipeline"""
        
        # Phase 1: Madugu routing with mode guardrails
        routing = self.madugu.route_query(user_input)
        
        # Phase 1.5: Mode guardrails override routing
        if mode == "creative":
            routing["quadrant"] = "south"  # Force creative/expressive domain
            routing["primary_agent"] = "TheStorykeeper" 
            routing["processing_mode"] = "creative"
        elif mode == "business":
            routing["quadrant"] = "north"  # Force business/structure domain
            routing["primary_agent"] = "TheClerk"
            routing["processing_mode"] = "business"
        # "auto" mode respects original Madugu routing
        
        # Phase 2: Memory context (if relevant)
        memory_context = []
        if len(user_input.split()) > 0:  # Search for any query
            memory_context = self.memory.search_memory(user_input, limit=3)
            if memory_context:
                print(f"[MEMORY] Found {len(memory_context)} relevant memories")
        
        # Phase 3: Create processing context
        processing_context = {
            "mode": mode,
            "routing": routing,
            "memory_context": [{"title": c.title, "body": c.body[:200]} for c in memory_context],
            "session_id": self.current_session,
            "node_id": self.node_id
        }
        
        # Phase 4: Generate response with MODEL-AGNOSTIC LLM routing
        try:
            from llm_init import init_llm, create_legacy_config
            
            # Initialize MODEL-AGNOSTIC LLM system
            if not hasattr(self, 'llm_client'):
                self.llm_client, self.chat_model, self.embed_model = init_llm()
                # Provider-neutral LLM client ready
                self.llm_config = create_legacy_config(self.llm_client, self.chat_model, self.embed_model)
                print(f"[AGNOSTIC] BrainBox initialized with {self.chat_model} via {self.llm_client.base_url}")
            
            # Generate response using intelligent routing
            messages = [
                {"role": "system", "content": f"You are {routing['primary_agent']}. Processing mode: {routing['processing_mode']}. Quadrant: {routing['quadrant']}."},
                {"role": "user", "content": user_input}
            ]
            llm_result = self.llm_client.chat(self.chat_model, messages)
            response = llm_result["choices"][0]["message"]["content"]
            
            # Add LLM routing info to result for transparency
            processing_context["llm_routing"] = {
                "selected_model": self.chat_model,
                "routing_reason": f"Agnostic routing to {routing['primary_agent']}", 
                "confidence": 1.0,
                "success": True
            }
            
        except ImportError as e:
            # Fallback to placeholder responses if LLM integration not available
            print(f"[ERROR] LLM IMPORT ERROR: {e}")
            print("Falling back to placeholder responses")
            if routing["processing_mode"] == "business":
                response = self._generate_business_response(user_input, processing_context)
            else:
                response = self._generate_creative_response(user_input, processing_context)
        except Exception as e:
            # Catch any other errors
            print(f"[ERROR] LLM ERROR: {e}")
            print("Falling back to placeholder responses")
            if routing["processing_mode"] == "business":
                response = self._generate_business_response(user_input, processing_context)
            else:
                response = self._generate_creative_response(user_input, processing_context)
            
            processing_context["llm_routing"] = {
                "selected_model": "placeholder",
                "routing_reason": "LLM integration not available",
                "confidence": 0.0,
                "success": False
            }
        
        # Phase 5: AXIOM review (business content only)
        # AXIOM only activates for business mode or auto mode with business routing
        should_axiom_review = (mode == "business") or (
            mode == "auto" and routing["processing_mode"] == "business"
        )
        
        if should_axiom_review:
            axiom_review = self.axiom.review_output(user_input, response, processing_context)
        else:
            # AXIOM stays out of personal/creative content
            axiom_review = {
                "axiom_active": False,
                "risks_flagged": [],
                "consequence_whisper": None,
                "business_appropriate": True
            }
        
        # Phase 6: Create memory card from interaction
        card = self._create_interaction_card(user_input, response, routing, axiom_review)
        
        result = {
            "response": response,
            "routing": routing,
            "axiom_review": axiom_review,
            "memory_card": card.id,
            "session_id": self.current_session,
            "node_id": self.node_id,
            "enterprise_mode": self.enterprise_mode,
            "memories_found": memory_context  # Add found memories to result
        }
        
        # Log complete interaction
        self.breakfast_chain.log_event("query_processed", "UnifiedBrainBox", {
            "input_length": len(user_input),
            "response_length": len(response),
            "axiom_active": axiom_review["axiom_active"],
            "risks_flagged": len(axiom_review["risks_flagged"])
        })
        
        return result
    
    def _generate_business_response(self, query: str, context: Dict) -> str:
        """Generate business-focused response (placeholder)"""
        # This would integrate with LLM
        return f"Business response for: {query[:50]}... (Routed through {context['routing']['quadrant']} quadrant)"
    
    def _generate_creative_response(self, query: str, context: Dict) -> str:
        """Generate creatively-processed response (placeholder)"""
        # This would integrate with LLM + emotional processing
        return f"Creative response for: {query[:50]}... (Processed by {context['routing']['primary_agent']})"
    
    def _create_interaction_card(self, user_input: str, response: str, routing: Dict, axiom_review: Dict) -> EnhancedCard:
        """Create memory card from interaction"""
        
        card_id = hashlib.sha256(f"{user_input}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        card = EnhancedCard(
            id=card_id,
            type="interaction",
            title=f"Query: {user_input[:50]}...",
            body=f"Input: {user_input}\n\nResponse: {response}",
            emotional_domain=EmotionalDomain(routing["emotion"]),
            quadrant=QuadrantDomain(routing["quadrant"]),
            voice_family=routing["primary_agent"],
            confidence_score=routing["intensity"],
            processing_mode=ProcessingMode(routing["processing_mode"]),
            axiom_reviewed=axiom_review["axiom_active"],
            session_id=self.current_session
        )
        
        return self.memory.create_card(card)
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        
        with sqlite3.connect(self.memory.db_path) as conn:
            cursor = conn.cursor()
            
            # Memory stats
            cursor.execute("SELECT COUNT(*) FROM memory_cards")
            total_cards = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM memory_cards WHERE axiom_reviewed = 1")
            axiom_reviewed = cursor.fetchone()[0]
        
        stats = {
            "node_id": self.node_id,
            "enterprise_mode": self.enterprise_mode,
            "session_id": self.current_session,
            "memory": {
                "total_cards": total_cards,
                "axiom_reviewed": axiom_reviewed,
                "storage_path": str(self.memory.db_path)
            },
            "breakfast_chain": {
                "verified": self.breakfast_chain.verify_chain(),
                "storage_path": str(self.breakfast_chain.db_path)
            },
            "components": ["Madugu", "AXIOM", "Memory", "BreakfastChain"]
        }
        
        return stats
    
    def export_session(self, session_id: Optional[str] = None) -> Path:
        """Export complete session with breakfast chain verification"""
        
        target_session = session_id or self.current_session
        
        export_data = {
            "session_id": target_session,
            "node_id": self.node_id,
            "enterprise_mode": self.enterprise_mode,
            "export_timestamp": datetime.now().isoformat(),
            "breakfast_chain_verified": self.breakfast_chain.verify_chain(),
            "system_stats": self.get_system_stats()
        }
        
        # Add breakfast chain events for session
        with sqlite3.connect(self.breakfast_chain.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM breakfast_chain WHERE session_id = ? ORDER BY timestamp",
                (target_session,)
            )
            
            export_data["breakfast_events"] = []
            for row in cursor.fetchall():
                export_data["breakfast_events"].append({
                    "timestamp": row[1],
                    "event_type": row[2],
                    "source": row[3],
                    "data": json.loads(row[4]),
                    "hash": row[7]
                })
        
        # Export to file
        export_path = self.data_dir / f"session_export_{target_session}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Log export
        self.breakfast_chain.log_event("session_exported", "UnifiedBrainBox", {
            "export_path": str(export_path),
            "session_id": target_session,
            "events_count": len(export_data["breakfast_events"])
        })
        
        return export_path

# Example usage and testing
if __name__ == "__main__":
    print(" Unified BrainBox System - Loading...")
    
    # Initialize system
    brain = UnifiedBrainBox()
    
    print(f" System initialized - Node ID: {brain.node_id}")
    print(f" System stats: {brain.get_system_stats()}")
    
    # Test queries
    test_queries = [
        ("Help me write marketing copy for Fort Workbook", "business"),
        ("I'm feeling frustrated with workplace issues", "creative"),
        ("Analyze the revenue potential of Mirror Protocol", "business"),
        ("Write a song about overcoming challenges", "creative")
    ]
    
    for query, mode in test_queries:
        print(f"\n Processing: {query}")
        result = brain.process_query(query, mode)
        
        print(f"    Routed to: {result['routing']['quadrant']} quadrant")
        print(f"    Emotion: {result['routing']['emotion']} ({result['routing']['intensity']:.2f})")
        print(f"    AXIOM active: {result['axiom_review']['axiom_active']}")
        if result['axiom_review']['risks_flagged']:
            print(f"    Risks: {[r['type'] for r in result['axiom_review']['risks_flagged']]}")
    
    # Test breakfast chain
    print(f"\n Breakfast chain verified: {brain.breakfast_chain.verify_chain()}")
    
    # Export session
    export_path = brain.export_session()
    print(f" Session exported to: {export_path}")
    
    print("\n✨ BrainBox ready for enterprise scaling!")