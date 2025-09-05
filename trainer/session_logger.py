#!/usr/bin/env python3
"""
Session Logger for SpineTrainer
===============================

Captures every user interaction with outcomes for training data generation.
Privacy-first: all data stays local, structured for learning.
"""

import json
import pathlib
from datetime import datetime
from typing import Dict, List, Any, Optional

class SessionLogger:
    """
    Logs user interactions for spine training
    Creates training data from real usage patterns
    """
    
    def __init__(self, data_dir: str = "brainbox_data"):
        self.data_dir = pathlib.Path(data_dir)
        self.sessions_dir = self.data_dir / "spine_sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_session_file(self) -> pathlib.Path:
        """Get today's session file"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.sessions_dir / f"{date_str}.jsonl"
        
    def log_interaction(self,
                       prompt: str,
                       intent: str,
                       intensity: float,
                       voices_used: List[str],
                       route_taken: str,
                       response_text: str,
                       success: bool,
                       user_feedback: Optional[str] = None,
                       latency_ms: Optional[int] = None,
                       spine_state: Optional[Dict] = None) -> str:
        """
        Log a complete user interaction for training
        
        Returns: session_id for referencing this interaction
        """
        
        session_id = f"{datetime.now().timestamp():.6f}"
        
        session_record = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "input": {
                "prompt": prompt,
                "prompt_length": len(prompt),
                "time_of_day": datetime.now().hour
            },
            "prediction": {
                "intent": intent,
                "intensity": intensity,
                "voices_selected": voices_used,
                "route": route_taken
            },
            "execution": {
                "response_length": len(response_text),
                "latency_ms": latency_ms or 0,
                "success": success
            },
            "outcome": {
                "user_feedback": user_feedback,
                "satisfaction_score": 1.0 if success else 0.0
            },
            "spine_context": spine_state or {}
        }
        
        # Append to today's session file
        session_file = self._get_session_file()
        with session_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(session_record, ensure_ascii=False) + "\n")
            
        return session_id
        
    def update_feedback(self, session_id: str, feedback_type: str, value: Any):
        """
        Update session with additional feedback
        (e.g., user clicked thumbs up later)
        """
        
        # For now, we'll store feedback updates in a separate file
        # Production version could update the original records
        
        feedback_file = self.sessions_dir / "feedback_updates.jsonl"
        update_record = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "feedback_type": feedback_type,
            "value": value
        }
        
        with feedback_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(update_record, ensure_ascii=False) + "\n")
            
    def load_recent_sessions(self, days: int = 7) -> List[Dict]:
        """
        Load sessions from the last N days for training
        """
        sessions = []
        
        # Get session files from last N days
        from datetime import datetime, timedelta
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            session_file = self.sessions_dir / f"{date_str}.jsonl"
            
            if session_file.exists():
                with session_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                session = json.loads(line)
                                sessions.append(session)
                            except json.JSONDecodeError:
                                continue
                                
        return sessions
        
    def get_training_stats(self, days: int = 7) -> Dict:
        """
        Get statistics about recent training data
        """
        sessions = self.load_recent_sessions(days)
        
        if not sessions:
            return {"total_sessions": 0, "message": "No training data available"}
            
        # Basic stats
        total_sessions = len(sessions)
        successful_sessions = sum(1 for s in sessions if s["execution"]["success"])
        success_rate = successful_sessions / total_sessions if total_sessions > 0 else 0
        
        # Intent distribution
        intent_counts = {}
        for session in sessions:
            intent = session["prediction"]["intent"]
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
        # Voice usage
        voice_counts = {}
        for session in sessions:
            for voice in session["prediction"]["voices_selected"]:
                voice_counts[voice] = voice_counts.get(voice, 0) + 1
                
        return {
            "total_sessions": total_sessions,
            "success_rate": success_rate,
            "intent_distribution": intent_counts,
            "voice_usage": voice_counts,
            "data_quality": "good" if total_sessions > 50 else "limited"
        }
        
    def export_training_data(self, days: int = 30, output_file: Optional[str] = None) -> str:
        """
        Export training data to a file for analysis or external training
        """
        sessions = self.load_recent_sessions(days)
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"training_export_{timestamp}.json"
            
        export_path = self.sessions_dir / output_file
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "days_included": days,
            "total_sessions": len(sessions),
            "sessions": sessions
        }
        
        with export_path.open("w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            
        return str(export_path)

# Simple CLI for testing
if __name__ == "__main__":
    import sys
    
    logger = SessionLogger()
    
    if len(sys.argv) < 2:
        print("Usage: python session_logger.py [stats|export|test]")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "stats":
        stats = logger.get_training_stats()
        print(json.dumps(stats, indent=2))
        
    elif command == "export":
        export_file = logger.export_training_data()
        print(f"Training data exported to: {export_file}")
        
    elif command == "test":
        # Log a test interaction
        session_id = logger.log_interaction(
            prompt="Help me plan my day",
            intent="planning",
            intensity=0.5,
            voices_used=["Planning Assistant"],
            route_taken="local_process",
            response_text="Here's a suggested daily plan...",
            success=True,
            user_feedback="helpful"
        )
        print(f"Test interaction logged: {session_id}")
        
        # Show stats
        stats = logger.get_training_stats()
        print("Current stats:", json.dumps(stats, indent=2))