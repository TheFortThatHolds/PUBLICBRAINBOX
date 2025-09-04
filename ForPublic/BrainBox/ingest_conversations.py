"""
Conversations.json Ingestion System
===================================

Loads conversations from Claude/ChatGPT exports into BrainBox memory.
Creates searchable memory cards with emotional tagging and semantic indexing.
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import re

from unified_brainbox import UnifiedBrainBox, EmotionalDomain, QuadrantDomain
from semantic_search import SemanticSearcher

class ConversationIngester:
    """Ingests conversation history into BrainBox memory"""
    
    def __init__(self, brainbox_data_path: Path = Path("./brainbox_data")):
        self.brain = UnifiedBrainBox(brainbox_data_path)
        self.searcher = SemanticSearcher()
        self.stats = {
            "messages_processed": 0,
            "memory_cards_created": 0,
            "conversations_loaded": 0
        }
    
    def detect_emotion(self, text: str) -> EmotionalDomain:
        """Simple emotion detection from text"""
        text_lower = text.lower()
        
        # Emotional keywords mapping
        if any(word in text_lower for word in ["angry", "frustrated", "pissed", "mad"]):
            return EmotionalDomain.ANGER
        elif any(word in text_lower for word in ["sad", "loss", "grief", "mourn"]):
            return EmotionalDomain.GRIEF
        elif any(word in text_lower for word in ["happy", "joy", "excited", "celebrate"]):
            return EmotionalDomain.JOY
        elif any(word in text_lower for word in ["scared", "fear", "afraid", "anxious"]):
            return EmotionalDomain.FEAR
        elif any(word in text_lower for word in ["love", "care", "heart", "affection"]):
            return EmotionalDomain.LOVE
        elif any(word in text_lower for word in ["business", "money", "profit", "revenue"]):
            return EmotionalDomain.BUSINESS
        else:
            return EmotionalDomain.PEACE  # Default
    
    def ingest_claude_export(self, json_path: Path):
        """Ingest Claude conversations export"""
        print(f"[INGEST] Loading Claude export from {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Claude export format has conversations as list
        conversations = data if isinstance(data, list) else data.get('conversations', [])
        
        for conv in conversations:
            self.stats["conversations_loaded"] += 1
            
            # Extract conversation metadata
            conv_id = conv.get('uuid', conv.get('id', 'unknown'))
            created_at = conv.get('created_at', datetime.now().isoformat())
            name = conv.get('name', 'Untitled Conversation')
            
            print(f"[INGEST] Processing conversation: {name}")
            
            # Process messages
            messages = conv.get('chat_messages', conv.get('messages', []))
            
            for msg in messages:
                self.stats["messages_processed"] += 1
                
                # Extract message content
                sender = msg.get('sender', msg.get('role', 'unknown'))
                text = msg.get('text', '')
                
                # Handle nested content structure
                if isinstance(text, list):
                    text = ' '.join([item.get('text', '') if isinstance(item, dict) else str(item) for item in text])
                elif isinstance(text, dict):
                    text = text.get('text', '')
                
                if not text or len(text) < 50:  # Skip very short messages
                    continue
                
                # Detect emotion
                emotion = self.detect_emotion(text)
                
                # Create memory card
                card_title = f"{name} - {sender}"
                card_body = text[:1000]  # Limit card body size
                
                # Store in memory
                card = self.brain.memory.create_memory_card(
                    title=card_title,
                    body=card_body,
                    emotional_domain=emotion,
                    source_file=str(json_path),
                    metadata={
                        "conversation_id": conv_id,
                        "sender": sender,
                        "created_at": created_at
                    }
                )
                
                self.stats["memory_cards_created"] += 1
                
                # Index for semantic search
                self.searcher.texts[f"conv_{conv_id}_{self.stats['messages_processed']}"] = text
                embedding = self.searcher.get_embedding(text[:2000])
                if embedding:
                    import numpy as np
                    self.searcher.embeddings[f"conv_{conv_id}_{self.stats['messages_processed']}"] = np.array(embedding)
        
        # Save semantic index
        self.searcher.save_index()
        
        print(f"[INGEST] Ingestion complete!")
        print(f"  - Conversations: {self.stats['conversations_loaded']}")
        print(f"  - Messages: {self.stats['messages_processed']}")
        print(f"  - Memory cards: {self.stats['memory_cards_created']}")
    
    def ingest_chatgpt_export(self, json_path: Path):
        """Ingest ChatGPT conversations export"""
        print(f"[INGEST] Loading ChatGPT export from {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # ChatGPT format varies but usually has conversations array
        for conv in data:
            self.stats["conversations_loaded"] += 1
            
            title = conv.get('title', 'Untitled')
            conv_id = conv.get('id', 'unknown')
            
            print(f"[INGEST] Processing conversation: {title}")
            
            # Navigate through the mapping structure
            mapping = conv.get('mapping', {})
            
            for node_id, node in mapping.items():
                message = node.get('message')
                if not message:
                    continue
                
                self.stats["messages_processed"] += 1
                
                # Extract content
                author_role = message.get('author', {}).get('role', 'unknown')
                content = message.get('content', {})
                
                # Handle content parts
                parts = content.get('parts', [])
                text = ' '.join([str(part) for part in parts if part])
                
                if not text or len(text) < 50:
                    continue
                
                # Process similar to Claude
                emotion = self.detect_emotion(text)
                
                card = self.brain.memory.create_memory_card(
                    title=f"{title} - {author_role}",
                    body=text[:1000],
                    emotional_domain=emotion,
                    source_file=str(json_path),
                    metadata={
                        "conversation_id": conv_id,
                        "node_id": node_id,
                        "role": author_role
                    }
                )
                
                self.stats["memory_cards_created"] += 1
                
                # Index for semantic search
                embedding = self.searcher.get_embedding(text[:2000])
                if embedding:
                    import numpy as np
                    self.searcher.embeddings[f"gpt_{conv_id}_{node_id}"] = np.array(embedding)
                    self.searcher.texts[f"gpt_{conv_id}_{node_id}"] = text
        
        # Save semantic index
        self.searcher.save_index()
        
        print(f"[INGEST] ChatGPT ingestion complete!")
        print(f"  - Conversations: {self.stats['conversations_loaded']}")
        print(f"  - Messages: {self.stats['messages_processed']}")
        print(f"  - Memory cards: {self.stats['memory_cards_created']}")
    
    def search_memories(self, query: str, limit: int = 5):
        """Search ingested memories semantically"""
        print(f"\n[SEARCH] Query: {query}")
        
        # Search semantic index
        results = self.searcher.search(query, top_k=limit)
        
        print(f"[SEARCH] Found {len(results)} relevant memories")
        
        for i, (key, text, similarity) in enumerate(results, 1):
            print(f"\n{i}. Similarity: {similarity:.3f}")
            print(f"   Key: {key}")
            print(f"   Preview: {text[:200]}...")
        
        return results

# Example usage
if __name__ == "__main__":
    ingester = ConversationIngester()
    
    # Look for conversation exports
    claude_export = Path(r"C:\Users\Fort That Holds LLC\OneDrive\Documents\GitHub\BrainBox\Conversations\conversations.json")
    chatgpt_export = Path("chatgpt_conversations.json")
    
    if claude_export.exists():
        print("[INGEST] Found Claude export")
        ingester.ingest_claude_export(claude_export)
    
    if chatgpt_export.exists():
        print("[INGEST] Found ChatGPT export")
        ingester.ingest_chatgpt_export(chatgpt_export)
    
    if not claude_export.exists() and not chatgpt_export.exists():
        print("[INGEST] No conversation exports found!")
        print("[INGEST] Place your conversations.json in the current directory")
    
    # Test search
    if ingester.stats["memory_cards_created"] > 0:
        print("\n" + "="*60)
        print("Testing memory search...")
        ingester.search_memories("emotional intelligence")
        ingester.search_memories("business strategy")