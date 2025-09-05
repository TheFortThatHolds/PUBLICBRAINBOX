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
        # Import BrainBox system and create interactive session
        from unified_brainbox import UnifiedBrainBox
        
        print("[BRAIN] BrainBox AI Spine Activated")
        print("=" * 40)
        
        # Initialize and start interactive session
        brainbox = UnifiedBrainBox()
        
        if len(sys.argv) > 1 and "--help" in sys.argv:
            print("BrainBox Commands:")
            print("  brainbox           - Start interactive session")
            print("  brainbox --help    - Show this help")
            print("  brainbox --status  - Show system status")
            return
            
        print("Type your query or 'exit' to quit...")
        while True:
            try:
                user_input = input("\n[YOU]: ").strip()
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    break
                if user_input:
                    response = brainbox.process_query(user_input)
                    print(f"\n[BRAIN]: {response}")
            except KeyboardInterrupt:
                break
        
    except KeyboardInterrupt:
        print("\n\n[BRAIN] BrainBox spine deactivated. Until next time...")
    except Exception as e:
        print(f"[ERROR] BrainBox initialization failed: {e}")
        print(f"Working directory: {os.getcwd()}")
        print("Try running from the BrainBox installation directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()