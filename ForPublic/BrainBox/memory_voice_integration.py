#!/usr/bin/env python3
"""
Memory + Voice Integration Prototype
Demonstrates voice-aware memory retrieval without external dependencies
Ships in 24 hours with real value
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import hashlib
from datetime import datetime
from unicode_sanitizer import sanitize_for_windows_terminal

class VoiceAwareMemory:
    """Minimal viable memory system with voice-specific retrieval patterns"""
    
    def __init__(self, db_path="brainbox_data/memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_search_tables()
        
        # Voice search personalities
        self.voice_patterns = {
            "The Archivist": {
                "weight_content": 0.8,
                "weight_title": 0.2, 
                "weight_recency": 0.1,
                "min_relevance": 0.3,
                "description": "Broad semantic search across all memory"
            },
            "Ming": {
                "weight_content": 0.5,
                "weight_title": 0.1,
                "weight_recency": 0.4,
                "min_relevance": 0.5,
                "description": "Emotional patterns and recent state"
            },
            "The Architect": {
                "weight_content": 0.3,
                "weight_title": 0.6,
                "weight_recency": 0.1,
                "min_relevance": 0.6,
                "description": "Technical precedents and plans"
            },
            "Camera Obscura": {
                "weight_content": 0.9,
                "weight_title": 0.1,
                "weight_recency": 0.0,
                "min_relevance": 0.4,
                "description": "Sensory and scene patterns"
            }
        }
        
    def _init_search_tables(self):
        """Create FTS5 tables for fast text search"""
        cursor = self.conn.cursor()
        
        # Check if FTS table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='cards_fts'
        """)
        
        if not cursor.fetchone():
            # Create FTS virtual table
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts 
                USING fts5(
                    card_id UNINDEXED,
                    title,
                    body,
                    emotional_domain,
                    voice_family,
                    tokenize='porter'
                )
            """)
            
            # Populate from existing cards
            cursor.execute("""
                INSERT INTO cards_fts (card_id, title, body, emotional_domain, voice_family)
                SELECT id, title, body, emotional_domain, voice_family FROM memory_cards
            """)
            
            self.conn.commit()
    
    def search_memory(self, query: str, voice: str = "The Archivist", limit: int = 10) -> List[Dict]:
        """
        Voice-aware memory search with personality-specific weighting
        """
        pattern = self.voice_patterns.get(voice, self.voice_patterns["The Archivist"])
        
        cursor = self.conn.cursor()
        
        # FTS5 search with BM25 ranking
        cursor.execute("""
            SELECT 
                c.id,
                c.title,
                c.body,
                c.created_at,
                c.emotional_domain,
                c.voice_family,
                bm25(cards_fts) as rank
            FROM cards_fts fts
            JOIN memory_cards c ON c.id = fts.card_id
            WHERE cards_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit * 2))  # Get extra for filtering
        
        results = []
        for row in cursor.fetchall():
            # Calculate voice-specific relevance
            relevance = self._calculate_relevance(
                row, query, pattern
            )
            
            if relevance >= pattern["min_relevance"]:
                # Sanitize content for Windows terminal
                body_text = (row["body"][:200] + "...") if row["body"] else "[No content]"
                results.append({
                    "id": row["id"],
                    "title": sanitize_for_windows_terminal(row["title"] or "[Untitled]"),
                    "content": sanitize_for_windows_terminal(body_text),
                    "relevance": relevance,
                    "created_at": row["created_at"],
                    "voice_perspective": f"{voice} sees this as relevant"
                })
        
        # Sort by voice-weighted relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:limit]
    
    def _calculate_relevance(self, row: sqlite3.Row, query: str, pattern: Dict) -> float:
        """Calculate voice-specific relevance score"""
        # Base BM25 score normalized
        base_score = abs(row["rank"]) / 10.0 if row["rank"] else 0.0
        
        # Title match bonus
        title_score = 1.0 if query.lower() in row["title"].lower() else 0.0
        
        # Recency score (newer = higher)
        if row["created_at"]:
            days_old = (datetime.now() - datetime.fromisoformat(row["created_at"])).days
            recency_score = max(0, 1.0 - (days_old / 365.0))
        else:
            recency_score = 0.5
        
        # Weighted combination
        final_score = (
            base_score * pattern["weight_content"] +
            title_score * pattern["weight_title"] +
            recency_score * pattern["weight_recency"]
        )
        
        return min(1.0, final_score)
    
    def disambiguate_voice(self, user_input: str) -> str:
        """Simple voice name disambiguation"""
        aliases = {
            "archivist": "The Archivist",
            "archive": "The Archivist",
            "ming": "Ming",
            "architect": "The Architect",
            "planner": "The Architect",
            "camera": "Camera Obscura",
            "obscura": "Camera Obscura",
            "visual": "Camera Obscura"
        }
        
        input_lower = user_input.lower()
        
        # Direct alias match
        if input_lower in aliases:
            return aliases[input_lower]
        
        # Partial match
        for alias, voice in aliases.items():
            if alias in input_lower or input_lower in alias:
                return voice
        
        # Default
        return "The Archivist"
    
    def demo_search(self, query: str):
        """Demo different voice perspectives on the same query"""
        print(f"\n[QUERY]: '{query}'\n")
        print("=" * 60)
        
        for voice in ["The Archivist", "Ming", "The Architect", "Camera Obscura"]:
            pattern = self.voice_patterns[voice]
            print(f"\n[VOICE]: {voice}")
            print(f"   {pattern['description']}")
            print("-" * 40)
            
            results = self.search_memory(query, voice, limit=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"   {i}. [{result['relevance']:.2f}] {result['title']}")
                    content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                    print(f"      {content_preview}")
            else:
                print("   No relevant memories from this perspective")
        
        print("\n" + "=" * 60)


def create_demo():
    """Create a shippable demo showing voice-aware memory"""
    print("""
====================================================================
   BRAINBOX MEMORY + VOICE INTEGRATION PROTOTYPE           
   Voice-Aware Memory Retrieval System                     
====================================================================
    """)
    
    mem = VoiceAwareMemory()
    
    # Demo 1: Same query, different voices
    print("\nDEMO 1: Multiple Voice Perspectives")
    mem.demo_search("consciousness")
    
    # Demo 2: Voice disambiguation
    print("\nDEMO 2: Voice Name Disambiguation")
    test_inputs = ["ming", "the archive guy", "camera"]
    for input_text in test_inputs:
        resolved = mem.disambiguate_voice(input_text)
        print(f"   '{input_text}' -> {resolved}")
    
    # Demo 3: Practical search
    print("\nDEMO 3: Practical Memory Search")
    try:
        query = input("\nEnter a search query (or press Enter for 'emotional regulation'): ")
    except (EOFError, KeyboardInterrupt):
        query = "emotional regulation"
    if not query:
        query = "emotional regulation"
    
    try:
        voice_input = input("Which voice perspective? (Enter for Archivist): ")
    except (EOFError, KeyboardInterrupt):
        voice_input = ""
    if voice_input:
        voice = mem.disambiguate_voice(voice_input)
    else:
        voice = "The Archivist"
    
    print(f"\nSearching as {voice}...")
    results = mem.search_memory(query, voice)
    
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. {result['title']} (relevance: {result['relevance']:.2f})")
        print(f"   {result['content']}")
        if result['created_at']:
            print(f"   Created: {result['created_at']}")
    
    print("""
====================================================================
   VALUE PROPOSITION                                        
   * Voice-aware search (not just keyword matching)        
   * Each voice has different relevance patterns           
   * Works with existing BrainBox memory.db                
   * No external dependencies, pure SQLite FTS5            
   * Shippable prototype in < 24 hours                     
====================================================================
    """)


if __name__ == "__main__":
    create_demo()