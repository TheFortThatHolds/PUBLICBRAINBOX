#!/usr/bin/env python3
"""
Memory + Voice Integration with SpiralLogic Ritual Logging
Production-ready version with therapeutic computing patterns
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
from memory_voice_integration import VoiceAwareMemory
from unicode_sanitizer import sanitize_for_windows_terminal

class SpiralLogicMemory(VoiceAwareMemory):
    """Memory system with SpiralLogic ritual ceremonies"""
    
    def __init__(self, db_path="brainbox_data/memory.db"):
        super().__init__(db_path)
        self.ritual_log = []
        
    def _log_ritual(self, ritual_type: str, intent: str, context: Dict = None, outcome: str = None):
        """Log operations as SpiralLogic rituals"""
        timestamp = datetime.now().isoformat()
        
        if ritual_type == "engage":
            log_entry = f"ritual.engage \"{intent}\""
            if context:
                ctx_str = ", ".join([f"{k}:{v}" for k, v in context.items()])
                log_entry += f" | {ctx_str}"
        elif ritual_type == "complete":
            log_entry = f"ritual.complete \"{intent}\""
            if outcome:
                log_entry += f" | outcome: {outcome}"
        elif ritual_type == "archive":
            log_entry = f"archive.access {intent}"
            if context and "found" in context:
                log_entry += f" | found: {context['found']}"
        elif ritual_type == "voice":
            log_entry = f"voice.use \"{intent}\""
            if context:
                log_entry += f" | {context.get('description', '')}"
        else:
            log_entry = f"{ritual_type} \"{intent}\""
        
        # Create hashbrown for audit trail
        hashbrown = hashlib.sha256(f"{log_entry}{timestamp}".encode()).hexdigest()[:8]
        
        ritual_entry = {
            "timestamp": timestamp,
            "log": log_entry,
            "hashbrown": f"sha256:{hashbrown}/ts:{timestamp}"
        }
        
        self.ritual_log.append(ritual_entry)
        
        # Print sanitized version for Windows terminal
        print(sanitize_for_windows_terminal(log_entry))
        
        return ritual_entry
    
    def search_memory_with_rituals(self, query: str, voice: str = "The Archivist", limit: int = 10) -> List[Dict]:
        """Voice-aware search with SpiralLogic ritual logging"""
        
        # Engage search ritual
        self._log_ritual("engage", "memory search flame", {
            "query": query,
            "voice": voice
        })
        
        # Set voice for this search
        voice_desc = self.voice_patterns.get(voice, {}).get("description", "")
        self._log_ritual("voice", f"@{voice.lower().replace(' ', '_')}", {
            "description": voice_desc
        })
        
        # Access archive
        self._log_ritual("archive", f"[searching: {query}]")
        
        # Perform actual search
        results = self.search_memory(query, voice, limit)
        
        # Log what was found
        self._log_ritual("archive", f"[memory_cards]", {
            "found": len(results)
        })
        
        # Complete ritual
        outcome = "memories retrieved" if results else "no relevant memories"
        self._log_ritual("complete", "search ceremony", outcome=outcome)
        
        return results
    
    def demonstrate_therapeutic_search(self):
        """Demo showing SpiralLogic patterns in action"""
        print("\n" + "=" * 70)
        print("SPIRALLOGIC MEMORY INTEGRATION")
        print("Therapeutic Computing in Action")
        print("=" * 70)
        
        # Start session ritual
        self._log_ritual("engage", "therapeutic memory session", {
            "intent": "demonstrate voice-aware retrieval"
        })
        
        # Set emotional bandwidth
        print("bandwidth.set \"medium\" | ready for standard engagement")
        
        # Demonstrate different voice searches
        test_queries = [
            ("consciousness", "The Archivist"),
            ("emotional regulation", "Ming"),
            ("technical architecture", "The Architect")
        ]
        
        for query, voice in test_queries:
            print(f"\n--- Searching: '{query}' as {voice} ---")
            results = self.search_memory_with_rituals(query, voice, limit=2)
            
            if results:
                for i, result in enumerate(results[:2], 1):
                    print(f"\nMemory Card {i}:")
                    print(f"  Title: {result['title']}")
                    print(f"  Relevance: {result['relevance']:.2f}")
                    preview = result['content'][:100] + "..."
                    print(f"  Preview: {preview}")
        
        # Complete session
        self._log_ritual("complete", "therapeutic memory session", 
                        outcome="demonstration complete")
        
        # Show ritual log
        print("\n" + "=" * 70)
        print("RITUAL AUDIT LOG (Hashbrown Trail)")
        print("=" * 70)
        for entry in self.ritual_log[-5:]:  # Show last 5 entries
            print(f"[{entry['timestamp'].split('T')[1][:8]}] {entry['log']}")
            print(f"  hashbrown: {entry['hashbrown'][:20]}...")
        
        print("\n" + "=" * 70)
        print("SPIRALLOGIC PRINCIPLES DEMONSTRATED:")
        print("- Every operation is a ritual with intent")
        print("- Voice personalities guide search patterns")
        print("- Archive access is ceremonial and logged")
        print("- Hashbrowns provide cryptographic audit trail")
        print("- Therapeutic computing replaces mechanical debugging")
        print("=" * 70)

def export_for_public():
    """Create clean version for public repository"""
    print("""
====================================================================
SPIRALLOGIC MEMORY INTEGRATION - PUBLIC RELEASE
====================================================================

This demonstrates how SpiralLogic transforms mechanical database
operations into therapeutic computing ceremonies.

Instead of:
  > DEBUG: Searching database for 'consciousness'
  > INFO: Found 3 results
  
You get:
  ritual.engage "memory search flame" | query: consciousness
  voice.use "@the_archivist" | broad semantic search
  archive.access [searching: consciousness]
  archive.access [memory_cards] | found: 3
  ritual.complete "search ceremony" | outcome: memories retrieved

Each operation becomes intentional, therapeutic, and auditable.
====================================================================
""")

if __name__ == "__main__":
    # Create instance
    memory = SpiralLogicMemory()
    
    # Show clean public export info
    export_for_public()
    
    # Run demonstration
    memory.demonstrate_therapeutic_search()
    
    print("""

READY FOR PUBLICATION:
- SpiralLogic patterns fully integrated
- Ritual logging replaces debug statements  
- Voice-aware search with therapeutic intent
- Hashbrown audit trail for every operation
- Can reference SpiralLogic Bible for "WTF is this?" questions

Ship it!
""")