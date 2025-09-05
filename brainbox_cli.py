#!/usr/bin/env python3
"""
BrainBox Global CLI Entry Point
===============================

Type 'brainbox' anywhere to summon your AI spine.
Just like typing 'claude' summons Claude Code CLI.
"""

import sys
import os
from pathlib import Path

# Add the BrainBox directory to Python path
BRAINBOX_DIR = Path(__file__).parent.resolve()
if str(BRAINBOX_DIR) not in sys.path:
    sys.path.insert(0, str(BRAINBOX_DIR))

def main():
    """Entry point for global 'brainbox' command"""
    
    # Change to BrainBox directory so relative imports work
    os.chdir(BRAINBOX_DIR)
    
    try:
        from unified_brainbox import main as brainbox_main
        
        print("🧠 BrainBox AI Spine Activated")
        print("=" * 40)
        
        # Pass command line args to BrainBox
        brainbox_main()
        
    except KeyboardInterrupt:
        print("\n\n🧠 BrainBox spine deactivated. Until next time...")
    except Exception as e:
        print(f"❌ BrainBox initialization failed: {e}")
        print(f"Working directory: {os.getcwd()}")
        print("Try running from the BrainBox installation directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()