#!/usr/bin/env python3
"""
BrainBox Easy Setup Script
Handles installation, configuration, and first-run setup
"""

import os
import sys
import subprocess
import json
from pathlib import Path

class BrainBoxSetup:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "brainbox_data"
        
    def check_python_version(self):
        """Ensure Python 3.8+ is installed"""
        if sys.version_info < (3, 8):
            print("❌ BrainBox requires Python 3.8 or higher")
            print(f"   Current version: {sys.version}")
            return False
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        return True
    
    def install_dependencies(self):
        """Install required packages"""
        print("\n📦 Installing dependencies...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True, capture_output=True)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def create_data_directories(self):
        """Set up data storage directories"""
        print("\n📁 Creating data directories...")
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "conversations").mkdir(exist_ok=True)
        (self.data_dir / "exports").mkdir(exist_ok=True)
        
        print("✅ Data directories created")
        return True
    
    def create_config_file(self):
        """Generate default configuration"""
        config_path = self.base_dir / "brainbox_config.json"
        
        if config_path.exists():
            print("⚠️  Configuration file already exists")
            return True
            
        default_config = {
            "llm_provider": "local",  # local, openai, anthropic
            "local_api_url": "http://localhost:1234",
            "memory_settings": {
                "max_memory_cards": 10000,
                "enable_voice_search": True,
                "default_voice": "The Archivist"
            },
            "privacy_settings": {
                "local_storage_only": True,
                "encrypt_memory_cards": False
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print("✅ Configuration file created")
        return True
    
    def test_installation(self):
        """Run basic functionality tests"""
        print("\n🧪 Testing installation...")
        
        try:
            # Test imports
            from memory_voice_integration import VoiceAwareMemory
            from unicode_sanitizer import sanitize_for_windows_terminal
            print("✅ Core modules import successfully")
            
            # Test database creation
            test_db = self.data_dir / "test.db"
            memory = VoiceAwareMemory(str(test_db))
            print("✅ Database creation works")
            
            # Clean up test
            if test_db.exists():
                test_db.unlink()
            
            return True
            
        except Exception as e:
            print(f"❌ Installation test failed: {e}")
            return False
    
    def show_next_steps(self):
        """Display usage instructions"""
        print("\n" + "="*60)
        print("🎉 BrainBox installation complete!")
        print("="*60)
        print("\nQuick Start:")
        print("1. Demo voice-aware search:")
        print("   python test_memory_voices.py")
        print("\n2. Try SpiralLogic integration:")
        print("   python memory_spirallogic_integration.py")
        print("\n3. Start main BrainBox:")
        print("   python unified_launcher.py")
        print("\n4. Read the documentation:")
        print("   MEMORY_INTEGRATION_README.md")
        print("\nConfiguration:")
        print(f"   Edit: {self.base_dir / 'brainbox_config.json'}")
        print(f"   Data: {self.data_dir}")
        print("\n" + "="*60)
    
    def run_setup(self):
        """Execute complete setup process"""
        print("🧠 BrainBox Setup Starting...")
        print("="*60)
        
        steps = [
            self.check_python_version,
            self.install_dependencies,
            self.create_data_directories,
            self.create_config_file,
            self.test_installation
        ]
        
        for step in steps:
            if not step():
                print("\n❌ Setup failed!")
                return False
        
        self.show_next_steps()
        return True

if __name__ == "__main__":
    setup = BrainBoxSetup()
    success = setup.run_setup()
    sys.exit(0 if success else 1)