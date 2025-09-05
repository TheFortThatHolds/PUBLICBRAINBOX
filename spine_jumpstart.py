#!/usr/bin/env python3
"""
BrainBox Spine Jumpstart Tool
=============================

Want to accelerate your spine's learning? This tool lets you:
- Unlock voices immediately based on your needs
- Set initial preferences to speed up learning
- Import voice patterns from other AI experiences

For full spine creation, upgrade to BrainBox Fort Kit Addon.
"""

import json, pathlib
from growing_spine_manager import GrowingSpine

class SpineJumpstart:
    """
    Accelerate your growing spine based on your known preferences
    """
    
    def __init__(self):
        self.spine = GrowingSpine()
        
    def quick_profile(self):
        """Fast setup to jumpstart your spine's learning"""
        print("🚀 BrainBox Spine Jumpstart")
        print("=" * 35)
        print("Let's accelerate your spine's learning based on what you already know about yourself.")
        print()
        
        # Quick assessment
        print("Quick questions to jumpstart your AI:")
        print()
        
        work_style = input("Your work style (focused/creative/collaborative/analytical): ").strip().lower()
        challenge_areas = input("What do you most need help with? (planning/creativity/decisions/perspective/grounding): ").strip().lower()
        ai_experience = input("Previous AI assistant experience (beginner/some/lots): ").strip().lower()
        
        print()
        print("🧠 Analyzing your profile...")
        
        # Determine jumpstart signals
        signals_to_boost = {}
        voices_to_unlock = []
        
        # Work style mapping
        if "creative" in work_style:
            signals_to_boost["creative_requests"] = 4  # Closer to unlocking Camera Obscura
            
        if "analytical" in work_style:
            signals_to_boost["emotional_processing"] = 0  # Analytical users may not need emotional processing yet
            
        # Challenge area mapping
        challenges = challenge_areas.split(",")
        for challenge in challenges:
            challenge = challenge.strip()
            if "creativity" in challenge or "visual" in challenge:
                signals_to_boost["creative_requests"] = signals_to_boost.get("creative_requests", 0) + 5
            if "grounding" in challenge or "overwhelm" in challenge or "emotional" in challenge:
                signals_to_boost["emotional_processing"] = signals_to_boost.get("emotional_processing", 0) + 6
                
        # Experience level
        if "lots" in ai_experience:
            signals_to_boost["cloud_requests"] = 3  # They might want cloud access sooner
            
        # Apply the jumpstart
        if signals_to_boost:
            print(f"🎯 Boosting your usage signals: {signals_to_boost}")
            
            # Load and update counters
            counters_file = pathlib.Path("brainbox_data/usage_signals.json")
            if counters_file.exists():
                counters = json.loads(counters_file.read_text())
            else:
                counters = {}
                
            for signal, boost in signals_to_boost.items():
                counters[signal] = counters.get(signal, 0) + boost
                
            counters_file.write_text(json.dumps(counters, indent=2))
            
            # Trigger an update to check for births
            result = self.spine.update_from_interaction(
                f"Jumpstart profile: {work_style}, needs help with: {challenge_areas}",
                "setup",
                ["The Navigator"],
                True
            )
            
            if result["voices_born"]:
                print(f"🌟 Immediately unlocked: {', '.join(result['voices_born'])}")
            else:
                print("📈 Signals boosted - voices will unlock as you use the system")
                
        else:
            print("🌱 Natural learning mode - voices will unlock as you interact")
            
        # Show current status
        status = self.spine.get_status_report()
        print(f"\n📊 Your spine status:")
        print(f"   Active voices: {len(status['active_voices'])}")
        print(f"   Total interactions: {status['total_interactions']}")
        
        ready_voices = [v for v in status['voices_ready_to_birth'] if v['ready']]
        if ready_voices:
            print(f"   🎉 Ready to unlock: {', '.join([v['name'] for v in ready_voices])}")
            
        print(f"\n✅ Your BrainBox spine is jumpstarted and ready to learn!")
        
    def voice_unlock_menu(self):
        """Let users unlock specific voices if they know they want them"""
        print("🔓 Voice Unlock Menu")
        print("=" * 25)
        
        status = self.spine.get_status_report()
        
        print("Available voices to unlock:")
        print()
        
        unlockable = []
        for i, voice_info in enumerate(status['voices_ready_to_birth'], 1):
            if not voice_info['ready']:
                print(f"{i}. {voice_info['name']} - {voice_info['description']}")
                print(f"   Progress: {voice_info['progress']}")
                unlockable.append(voice_info)
            else:
                print(f"{i}. {voice_info['name']} - READY TO UNLOCK! ⭐")
                unlockable.append(voice_info)
        
        if not unlockable:
            print("All voices are already unlocked!")
            return
            
        print(f"\n{len(unlockable) + 1}. Skip - let voices unlock naturally")
        
        choice = input(f"\nSelect voice to unlock (1-{len(unlockable) + 1}): ").strip()
        
        try:
            choice_idx = int(choice) - 1
            if choice_idx == len(unlockable):
                print("🌱 Natural learning mode selected")
                return
                
            if 0 <= choice_idx < len(unlockable):
                voice_info = unlockable[choice_idx]
                voice_name = voice_info['name']
                
                if voice_info['ready']:
                    # Just trigger an update to birth it
                    result = self.spine.update_from_interaction(
                        f"Manual unlock requested for {voice_name}",
                        "unlock",
                        ["The Navigator"], 
                        True
                    )
                    print(f"🌟 {voice_name} unlocked!")
                else:
                    # Boost signals to unlock it
                    print(f"🚀 Boosting signals to unlock {voice_name}...")
                    
                    counters_file = pathlib.Path("brainbox_data/usage_signals.json")
                    counters = json.loads(counters_file.read_text()) if counters_file.exists() else {}
                    
                    # Get the spine data to find birth criteria
                    spine_file = pathlib.Path("brainbox_data/growing_spine.json")
                    spine_data = json.loads(spine_file.read_text())
                    
                    voice_config = spine_data["latent_voices"][voice_name]
                    criteria = voice_config["birth_criteria"]
                    
                    # Boost signals to meet criteria
                    for signal, threshold in criteria.items():
                        counters[signal] = max(counters.get(signal, 0), threshold)
                        
                    counters_file.write_text(json.dumps(counters, indent=2))
                    
                    # Trigger update
                    result = self.spine.update_from_interaction(
                        f"Manual unlock requested for {voice_name}",
                        "unlock",
                        ["The Navigator"],
                        True
                    )
                    
                    if voice_name in result["voices_born"]:
                        print(f"🌟 {voice_name} successfully unlocked!")
                    else:
                        print(f"❌ Unlock failed - please try again")
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")

def main():
    """Spine jumpstart menu"""
    jumpstart = SpineJumpstart()
    
    print("🧠 BrainBox Spine Jumpstart Tools")
    print("=" * 40)
    print()
    print("Choose your jumpstart method:")
    print("1. Quick Profile (recommended for new users)")
    print("2. Manual Voice Unlock")
    print("3. Show Current Status")
    print("4. Natural Learning (no jumpstart)")
    
    choice = input("\nSelect (1-4): ").strip()
    
    if choice == "1":
        jumpstart.quick_profile()
    elif choice == "2":
        jumpstart.voice_unlock_menu()
    elif choice == "3":
        status = jumpstart.spine.get_status_report()
        print(json.dumps(status, indent=2))
    elif choice == "4":
        print("🌱 Natural learning mode - your spine will grow with usage")
    else:
        print("Invalid choice")
        
    print(f"\n💡 For advanced voice creation, check out BrainBox Fort Kit Addon!")

if __name__ == "__main__":
    main()