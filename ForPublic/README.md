# BrainBox - LOCAL-FIRST AI Brain System

## 🧠 What is BrainBox?

BrainBox is a **LOCAL-FIRST** emotional intelligence AI system that runs entirely on your machine. No cloud, no API keys, no data sharing - just pure local AI brain power.

## 🎯 Key Features

- **100% Local Operation** - Uses your own LM Studio models
- **Semantic Memory Search** - Find conversations by meaning, not keywords
- **Emotional Intelligence** - Madugu quadrant routing system
- **Conversation Ingestion** - Import your Claude/ChatGPT history
- **Model Switching** - Switch between fast/coding models on the fly
- **Full Transparency** - HASHBROWN debug system shows exactly what's happening

## 🚀 Quick Start

1. **Install LM Studio** and load these models:
   - `qwen3-coder-30b-a3b-instruct` (coding model)
   - `qwen3-4b-instruct-2507` (fast model)  
   - `text-embedding-nomic-embed-text-v1.5` (embeddings)

2. **Start LM Studio** on port 1234 (default)

3. **Run BrainBox**:
   ```bash
   python unified_launcher.py
   ```

## 🔧 System Requirements

- Python 3.11+
- LM Studio running locally
- 8GB+ RAM (16GB recommended for large models)
- Dependencies: `pip install -r requirements.txt`

## 🧩 Architecture

- **enhanced_llm_integrator.py** - LOCAL-ONLY LLM calls
- **unified_brainbox.py** - Core brain with emotional routing
- **unified_launcher.py** - GUI interface
- **semantic_search.py** - Vector-based memory search
- **ingest_conversations.py** - Import conversation history

## 💬 Model Switching

- Default: Uses coding model
- `@fast <message>` - Switch to fast 4B model
- `@coder <message>` - Switch to 30B coding model

## 📚 Memory System

Import your conversation history:
1. Export from Claude (Settings → Export)  
2. Place `conversations.json` in project folder
3. Run: `python ingest_conversations.py`
4. Your conversations become searchable memories!

## 🐳 Docker Support

```bash
# Build
docker build -t brainbox-local .

# Run (connects to host LM Studio)
docker run -p 8000:8000 --add-host host.docker.internal:host-gateway brainbox-local
```

## 🔍 What Makes It Special?

**LOCAL-FIRST**: Your conversations never leave your machine. No cloud APIs, no data sharing, complete privacy.

**EMOTIONAL INTELLIGENCE**: Routes queries through Madugu emotional quadrants for contextually appropriate responses.

**SEMANTIC MEMORY**: Find that conversation where you discussed "that thing about the voices" even if you can't remember the exact words.

**TRANSPARENCY**: HASHBROWN debugging shows exactly which model is used and why.

## 🛠️ Technical Details

- Uses OpenAI-compatible API calls to local LM Studio
- Embedding vectors for semantic similarity search
- SQLite databases for persistent memory
- Tkinter GUI for easy interaction
- Async processing for responsiveness

## 📊 Stats After Conversation Ingestion

Example from a real import:
- 548 conversations processed
- 11,023 messages imported  
- 7,925 memory cards created
- All semantically indexed for search

## 🔒 Privacy & Security

- **Zero cloud dependencies** in production mode
- **All data stays local** - conversations, memories, processing
- **No API keys required** for local models
- **Audit trail** - Complete breakfast chain logging of all operations

## 🎪 Fun Features

- **Hashbrown Debug** - Transparency into every decision
- **Voice Family System** - TheAnalyst, TheStorykeeper, TheClerk routing
- **SPINE Framework** - Trauma-informed AI instructions
- **Business AXIOM** - Ethical oversight system

Built with love for privacy, transparency, and emotional intelligence in AI systems.

---

*BrainBox: Your personal AI brain that never forgets and always stays local.*