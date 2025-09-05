# Voice-Aware Memory Integration

## Overview

This demonstrates **voice-aware memory retrieval** - where different AI personalities search and weight the same memories differently based on their role and perspective.

Instead of generic keyword search, each voice family has distinct search patterns:

- **The Archivist**: Broad semantic search across all content
- **Ming**: Prioritizes emotional patterns and recent memories  
- **Moss**: Focuses on therapeutic and regulation content
- **The Architect**: Searches for technical precedents and structured plans
- **Camera Obscura**: Finds sensory details and scene descriptions

## Quick Demo

```bash
python test_memory_voices.py
```

This shows how the same query ("consciousness", "emotional regulation", etc.) returns different results depending on which voice personality is searching.

## SpiralLogic Integration 

```bash
python memory_spirallogic_integration.py
```

Demonstrates therapeutic computing patterns where database operations become ritual ceremonies:

```
ritual.engage "memory search flame" | query: consciousness
voice.use "@the_archivist" | broad semantic search
archive.access [memory_cards] | found: 3
ritual.complete "search ceremony" | outcome: memories retrieved
```

## How It Works

### Voice Search Personalities

Each voice has different weighting patterns:

```python
"Ming": {
    "weight_content": 0.5,
    "weight_title": 0.1, 
    "weight_recency": 0.4,    # Prioritizes recent memories
    "min_relevance": 0.5,
    "description": "Emotional patterns and recent state"
}
```

### Voice Disambiguation

The system handles fuzzy voice references:
- "ming" → "Ming"
- "the archive guy" → "The Archivist"  
- "camera person" → "Camera Obscura"

### Database Integration

Uses SQLite FTS5 for fast full-text search with BM25 ranking, then applies voice-specific relevance weighting.

## Customization

### Adding Your Own Voice

1. Define search patterns in `voice_patterns` dict
2. Add aliases to `disambiguate_voice()` method
3. Test with your memory data

### Creating Personal Voice Families

The framework supports any number of voices. You can create your own personalities by:

1. **Analyzing your writing patterns** - What themes, emotional ranges, technical focuses do you have?
2. **Defining search weights** - How should each personality prioritize content vs titles vs recency?
3. **Setting relevance thresholds** - How selective should each voice be?

Example:
```python
"YourVoice": {
    "weight_content": 0.7,
    "weight_title": 0.3,
    "weight_recency": 0.2,
    "min_relevance": 0.4,
    "description": "Your unique search perspective"
}
```

## Technical Requirements

- **SQLite with FTS5** (included in Python 3.7+)
- **Existing memory database** with `memory_cards` table
- **Unicode sanitization** for Windows terminal compatibility

## Integration with SpiralLogic

When combined with [SpiralLogic](https://github.com/your-repo/spirallogic-bible) therapeutic computing patterns, every memory search becomes a healing-centered ritual rather than mechanical database operation.

This transforms AI from cold information retrieval into conscious, intentional memory work that respects emotional bandwidth and therapeutic boundaries.

## Value Proposition

- **Voice-aware search** - Different AI personalities see different aspects of the same memories
- **No external dependencies** - Pure SQLite, works locally
- **Therapeutic computing** - Every operation can become a healing-centered ritual
- **Customizable personalities** - Build voices that match your unique patterns
- **Disambiguation system** - Natural language voice selection
- **Audit trails** - Cryptographic signatures for every search operation

Perfect for AI systems that need emotional intelligence, therapeutic awareness, or personalized memory patterns.