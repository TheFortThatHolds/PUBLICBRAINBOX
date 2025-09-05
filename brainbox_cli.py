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
from unicode_sanitizer import sanitize_for_windows_terminal

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
        
        if len(sys.argv) > 1:
            if "--help" in sys.argv:
                print("BrainBox Commands:")
                print("  brainbox           - Start interactive session")
                print("  brainbox --help    - Show this help")
                print("  brainbox --status  - Show system status")
                return
            elif "--status" in sys.argv:
                print(f"[BRAIN STATUS] System initialized successfully")
                print(f"[BRAIN STATUS] Working directory: {os.getcwd()}")
                print(f"[BRAIN STATUS] BrainBox ready for queries")
                return
            
        print("Type your query or 'exit' to quit...")
        while True:
            try:
                user_input = input("\n[YOU]: ").strip()
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    break
                if user_input:
                    response_dict = brainbox.process_query(user_input)
                    # Extract the actual response text
                    response_text = response_dict.get('response', str(response_dict))
                    # Sanitize response for Windows terminal
                    safe_response = sanitize_for_windows_terminal(response_text)
                    print(f"\n[BRAIN]: {safe_response}")
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