#!/usr/bin/env python3
"""
BrainBox Setup Script
====================

Quick setup for BrainBox LOCAL-FIRST AI System
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    return True

def check_lm_studio():
    """Check if LM Studio is running"""
    print("🔍 Checking LM Studio connection...")
    try:
        import requests
        response = requests.get("http://10.14.0.2:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("✅ LM Studio is running!")
            print(f"   Available models: {len(models.get('data', []))}")
            return True
        else:
            print("❌ LM Studio not responding correctly")
            return False
    except Exception as e:
        print("❌ LM Studio not running or not accessible")
        print(f"   Error: {e}")
        return False

def create_data_dir():
    """Create brainbox_data directory"""
    data_dir = Path("brainbox_data")
    data_dir.mkdir(exist_ok=True)
    print(f"📁 Created data directory: {data_dir.absolute()}")

def main():
    """Main setup process"""
    print("🧠 BrainBox Setup")
    print("==================")
    
    # Step 1: Install dependencies
    if not install_requirements():
        print("\n❌ Setup failed - could not install dependencies")
        return
    
    # Step 2: Check LM Studio
    if not check_lm_studio():
        print("\n⚠️  LM Studio not detected!")
        print("   Make sure LM Studio is running with a model loaded")
        print("   Recommended models:")
        print("   - qwen3-coder-30b-a3b-instruct (coding)")
        print("   - qwen3-4b-instruct-2507 (fast)")
        print("   - text-embedding-nomic-embed-text-v1.5 (embeddings)")
    
    # Step 3: Create directories
    create_data_dir()
    
    print("\n🎉 BrainBox setup complete!")
    print("\n🚀 To start BrainBox:")
    print("   python unified_launcher.py")
    print("\n📚 To import conversations:")
    print("   1. Place conversations.json in this directory")
    print("   2. Run: python ingest_conversations.py")

if __name__ == "__main__":
    main()