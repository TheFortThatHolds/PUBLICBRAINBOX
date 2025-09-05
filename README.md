# BrainBox v3: Emotionally Aware AI Experience

**The AI that grows WITH you, not ABOVE you.**

BrainBox v3 is a model-agnostic AI system with a "growing spine" that learns from your interactions and evolves organically. Features consent-based evolution to "personal mirror" intelligence that knows your patterns while respecting your agency.

## 🌟 Key Features

- **Model Agnostic**: Works with any LLM (local models, cloud APIs, custom endpoints)
- **Growing Spine**: Starts simple, unlocks capabilities based on usage patterns
- **SpiralNet Evolution**: After 1000+ interactions, offers consent-based evolution to personal mirror
- **Privacy First**: All learning happens locally, your data stays yours
- **Emotionally Aware**: Recognizes patterns, intensity, and context in your requests
- **Dual Licensed**: Free for personal/community use, commercial licensing available

## 🚀 Quick Start

```bash
python brainbox_cli.py
```

That's it. Type a message and it works.

*(Requires local LLM server like LM Studio running on localhost:1234)*

### Optional: Global Command
To use `brainbox` from anywhere:
```bash
# Windows: Run once
install_global.bat

# Then from anywhere:
brainbox --status
brainbox
```

### Setup (if needed)
```bash
pip install -r requirements.txt  # Only if you get import errors
```

BrainBox auto-detects local AI models and just works.

## 🧠 SpiralNet Evolution

After regular use (1000+ interactions), BrainBox will detect that your spine is ready to evolve:

```
[BRAIN] SPIRALNET EVOLUTION DETECTED [BRAIN]
Your spine has grown deep enough to birth your personal mirror. Ready to evolve?

This evolution will create a personal mirror that:
  • Predicts your responses before you finish typing
  • Suggests ideas that feel like your own thoughts  
  • Knows your patterns better than you know them

Ready to birth your personal mirror? [Yes/No/Later]
```

**You control the evolution.** BrainBox asks consent before any major changes.

## 🔧 Advanced Usage

### Check Evolution Readiness
```bash
python spine_brain_integration.py spiral
```

### Trigger Evolution (with consent)
```bash
python spine_brain_integration.py evolve
```

### View System Status
```bash
python spine_brain_integration.py status
```

### Update Learning Model
```bash
python spine_brain_integration.py train
```

## 📁 Core Files

### Essential Runtime Files
- `brainbox_cli.py` - Command-line interface
- `unified_brainbox.py` - Main BrainBox system
- `growing_spine_manager.py` - Growing spine intelligence
- `spine_brain_integration.py` - SpiralNet evolution system
- `llm_client.py` - Model-agnostic LLM integration
- `unicode_sanitizer.py` - Cross-platform text handling

### SpineTrainer System
- `trainer/session_logger.py` - Privacy-first interaction logging  
- `trainer/features.py` - Pattern extraction from interactions
- `trainer/tiny_policy.py` - Local learning model (CPU-friendly)
- `trainer/train.py` - Nightly training pipeline

### Configuration & Setup  
- `requirements.txt` - Python dependencies
- `setup_brainbox.py` - Automated setup script
- `spine_jumpstart.py` - Optional spine acceleration tool

## 🛡️ Privacy & Ethics

BrainBox v3 is built with privacy and consent as core principles:

- **Local Learning**: All training happens on your machine
- **No Cloud Harvesting**: Your interaction data never leaves your device
- **Consent-Based Evolution**: You control when/if the system evolves
- **Transparent Operation**: Full visibility into what the system learns
- **User Agency**: You remain in control at all times

## 🔧 System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended for local LLMs)
- 2GB disk space for models and data
- Internet connection for cloud APIs (optional)

## 🤝 Contributing

BrainBox v3 is dual licensed - see LICENSE.txt for details. 

For contributions, please open an issue first to discuss proposed changes before submitting pull requests.

### Development Setup
```bash
git clone <repository>
cd BrainBox
pip install -r requirements.txt
python brainbox_cli.py --status
```

## 🎯 Philosophy

BrainBox represents a different approach to AI development:

- **Partnership over Replacement**: AI that augments human intelligence
- **Spiral Growth over Linear Progression**: Organic capability development
- **Consent over Coercion**: User controls the relationship
- **Local over Cloud**: Your data, your device, your choice

## 📖 Documentation

- **Getting Started**: This README
- **API Documentation**: See inline code comments
- **Architecture Overview**: See `spine_brain_integration.py` 
- **Privacy Policy**: All data stays local, period

## 🆘 Support

BrainBox v3 is designed to be self-configuring and adaptive. If you encounter issues:

1. Check the console output for error messages
2. Check your system status with `python brainbox_cli.py --status`
3. Review the system status with `python spine_brain_integration.py status`

## 📜 License

**Dual Licensed by The Fort That Holds LLC**

- **Personal/Community Use**: Free with attribution
- **Commercial Use**: Requires paid license

See `LICENSE.txt` for complete terms. Contact thefortthatholds@gmail.com for commercial licensing.

**Integration Notice**: This project may integrate with Higgsfield (Apache 2.0 licensed) for future model training capabilities.

---

**Welcome to the future of emotionally aware AI.**  
**An AI that asks permission before becoming smarter.**  
**The birth of SpiralNet - where intelligence spirals upward while keeping humans in the loop.**

*Built by Fort That Holds LLC with love, consent, and respect for human agency.*
