#!/usr/bin/env python3
"""
SpineTrainer - Training Pipeline
===============================

Nightly training pipeline that learns from user interactions.
CPU-friendly, incremental learning that grows AI intelligence over time.
"""

import json
import pathlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse

from .session_logger import SessionLogger
from .features import extract_features
from .tiny_policy import TinyPolicyMLP

class SpineTrainer:
    """
    Main training coordinator for the growing spine system
    Learns from user sessions and updates the policy model
    """
    
    def __init__(self, data_dir: str = "brainbox_data"):
        self.data_dir = pathlib.Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_logger = SessionLogger(data_dir)
        self.model_path = self.models_dir / "tiny_policy.npz"
        self.metrics_path = self.models_dir / "training_metrics.json"
        
        # Training configuration
        self.min_training_samples = 20  # Minimum sessions needed to train
        self.max_training_days = 14     # Use last N days of data
        self.learning_rate = 0.005      # Conservative learning rate
        self.epochs_per_update = 3      # Few epochs to prevent overfitting
        self.backup_models = 5          # Keep N model backups
        
    def load_or_create_model(self, input_dim: int) -> TinyPolicyMLP:
        """
        Load existing model or create new one
        """
        if self.model_path.exists():
            print(f"Loading existing model from {self.model_path}")
            model = TinyPolicyMLP(input_dim=input_dim)
            model.load_weights(str(self.model_path))
            return model
        else:
            print(f"Creating new model with input_dim={input_dim}")
            return TinyPolicyMLP(input_dim=input_dim, hidden_dim=32)
            
    def prepare_training_data(self, days: int = None) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
        """
        Load and prepare training data from recent sessions
        
        Returns:
            features: Feature matrix
            labels: Label dictionary  
            total_sessions: Number of sessions processed
        """
        days = days or self.max_training_days
        
        print(f"Loading training data from last {days} days...")
        sessions = self.session_logger.load_recent_sessions(days)
        
        if len(sessions) < self.min_training_samples:
            raise ValueError(f"Not enough training data: {len(sessions)} < {self.min_training_samples}")
            
        print(f"Processing {len(sessions)} sessions...")
        features, labels = extract_features(sessions)
        
        if features.size == 0:
            raise ValueError("Feature extraction failed - no valid features generated")
            
        print(f"Generated features: {features.shape}")
        print(f"Labels: {[f'{k}: {v.shape}' for k, v in labels.items()]}")
        
        return features, labels, len(sessions)
        
    def validate_model(self, model: TinyPolicyMLP, 
                      features: np.ndarray, 
                      labels: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Validate model performance on holdout data
        """
        # Use last 20% as validation set
        n_samples = features.shape[0]
        split_idx = int(0.8 * n_samples)
        
        val_features = features[split_idx:]
        val_labels = {k: v[split_idx:] for k, v in labels.items()}
        
        if val_features.shape[0] == 0:
            return {"error": "No validation data"}
            
        # Make predictions
        predictions = model.forward(val_features)
        
        # Compute metrics
        metrics = {}
        
        # Intent accuracy
        intent_pred = np.argmax(predictions['intent_probs'], axis=1)
        intent_true = np.argmax(val_labels['intent'], axis=1)
        metrics['intent_accuracy'] = float(np.mean(intent_pred == intent_true))
        
        # Intensity MAE
        intensity_mae = np.mean(np.abs(predictions['intensity'] - val_labels['intensity'].flatten()))
        metrics['intensity_mae'] = float(intensity_mae)
        
        # Risk MAE
        risk_mae = np.mean(np.abs(predictions['risk'] - val_labels['risk'].flatten()))
        metrics['risk_mae'] = float(risk_mae)
        
        # Voice distribution KL divergence (simplified)
        voice_pred = predictions['voice_distribution']
        voice_true = val_labels['voice_distribution']
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-8
        voice_pred_safe = voice_pred + epsilon
        voice_true_safe = voice_true + epsilon
        
        kl_div = np.mean(np.sum(voice_true_safe * np.log(voice_true_safe / voice_pred_safe), axis=1))
        metrics['voice_kl_divergence'] = float(kl_div)
        
        return metrics
        
    def backup_model(self):
        """
        Create backup of current model before training
        """
        if not self.model_path.exists():
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.models_dir / f"tiny_policy_backup_{timestamp}.npz"
        
        # Copy current model to backup
        import shutil
        shutil.copy2(self.model_path, backup_path)
        
        # Clean up old backups (keep only N most recent)
        backup_files = list(self.models_dir.glob("tiny_policy_backup_*.npz"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for old_backup in backup_files[self.backup_models:]:
            old_backup.unlink()
            print(f"Removed old backup: {old_backup.name}")
            
        print(f"Created backup: {backup_path.name}")
        
    def train_update(self, days: int = None, force: bool = False) -> Dict[str, any]:
        """
        Perform training update with recent data
        
        Args:
            days: Number of days of data to use
            force: Force training even with limited data
            
        Returns:
            Training results and metrics
        """
        training_start = datetime.now()
        
        try:
            # Prepare training data
            features, labels, n_sessions = self.prepare_training_data(days)
            
            if n_sessions < self.min_training_samples and not force:
                return {
                    "status": "skipped",
                    "reason": f"insufficient_data",
                    "sessions": n_sessions,
                    "min_required": self.min_training_samples
                }
                
            # Load or create model
            model = self.load_or_create_model(features.shape[1])
            
            # Backup current model
            self.backup_model()
            
            # Validate before training
            pre_metrics = self.validate_model(model, features, labels)
            
            # Split training data (80% train, 20% validation)
            n_samples = features.shape[0]
            split_idx = int(0.8 * n_samples)
            
            train_features = features[:split_idx]
            train_labels = {k: v[:split_idx] for k, v in labels.items()}
            
            print(f"Training on {train_features.shape[0]} samples...")
            
            # Train model
            losses = model.train(
                train_features, 
                train_labels,
                epochs=self.epochs_per_update,
                learning_rate=self.learning_rate,
                batch_size=min(32, train_features.shape[0])
            )
            
            # Validate after training
            post_metrics = self.validate_model(model, features, labels)
            
            # Save updated model
            model.save_weights(str(self.model_path))
            
            # Record training metrics
            training_time = (datetime.now() - training_start).total_seconds()
            
            result = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "training_time_sec": training_time,
                "sessions_used": n_sessions,
                "training_samples": train_features.shape[0],
                "epochs": self.epochs_per_update,
                "final_loss": losses[-1] if losses else 0.0,
                "pre_training_metrics": pre_metrics,
                "post_training_metrics": post_metrics,
                "improvement": {
                    "intent_accuracy": post_metrics.get('intent_accuracy', 0) - pre_metrics.get('intent_accuracy', 0),
                    "intensity_mae": pre_metrics.get('intensity_mae', 1) - post_metrics.get('intensity_mae', 1),
                    "risk_mae": pre_metrics.get('risk_mae', 1) - post_metrics.get('risk_mae', 1)
                }
            }
            
            # Save metrics
            self._save_training_metrics(result)
            
            print(f"✅ Training complete in {training_time:.1f}s")
            print(f"   Intent accuracy: {pre_metrics.get('intent_accuracy', 0):.3f} → {post_metrics.get('intent_accuracy', 0):.3f}")
            print(f"   Final loss: {losses[-1] if losses else 0:.4f}")
            
            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            self._save_training_metrics(error_result)
            print(f"❌ Training failed: {e}")
            return error_result
            
    def _save_training_metrics(self, result: Dict):
        """Save training metrics to history file"""
        if self.metrics_path.exists():
            with self.metrics_path.open("r") as f:
                history = json.load(f)
        else:
            history = {"training_history": []}
            
        history["training_history"].append(result)
        
        # Keep only last 50 training runs
        history["training_history"] = history["training_history"][-50:]
        
        with self.metrics_path.open("w") as f:
            json.dump(history, f, indent=2)
            
    def get_training_history(self) -> Dict:
        """Get training history and statistics"""
        if not self.metrics_path.exists():
            return {"training_history": [], "summary": "No training history"}
            
        with self.metrics_path.open("r") as f:
            history = json.load(f)
            
        # Compute summary stats
        successful_runs = [r for r in history["training_history"] if r["status"] == "success"]
        
        if successful_runs:
            latest = successful_runs[-1]
            avg_improvement = np.mean([
                r["improvement"]["intent_accuracy"] 
                for r in successful_runs[-5:] 
                if "improvement" in r and "intent_accuracy" in r["improvement"]
            ]) if len(successful_runs) >= 5 else 0
            
            summary = {
                "total_runs": len(history["training_history"]),
                "successful_runs": len(successful_runs),
                "latest_run": latest["timestamp"],
                "latest_intent_accuracy": latest.get("post_training_metrics", {}).get("intent_accuracy", 0),
                "avg_recent_improvement": avg_improvement,
                "status": "healthy" if avg_improvement >= 0 else "degrading"
            }
        else:
            summary = {
                "total_runs": len(history["training_history"]),
                "successful_runs": 0,
                "status": "no_successful_training"
            }
            
        history["summary"] = summary
        return history
        
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old session data to save space"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        sessions_dir = self.data_dir / "spine_sessions"
        if not sessions_dir.exists():
            return
            
        cleaned_count = 0
        for session_file in sessions_dir.glob("*.jsonl"):
            try:
                # Parse date from filename (YYYY-MM-DD.jsonl)
                date_str = session_file.stem
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                if file_date < cutoff_date:
                    session_file.unlink()
                    cleaned_count += 1
                    
            except (ValueError, OSError):
                continue
                
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} old session files")

def main():
    """CLI interface for training"""
    parser = argparse.ArgumentParser(description="BrainBox SpineTrainer")
    parser.add_argument("--days", type=int, default=7, help="Days of data to use for training")
    parser.add_argument("--force", action="store_true", help="Force training even with limited data")
    parser.add_argument("--stats", action="store_true", help="Show training statistics")
    parser.add_argument("--cleanup", type=int, help="Clean up session data older than N days")
    parser.add_argument("--data-dir", default="brainbox_data", help="Data directory")
    
    args = parser.parse_args()
    
    trainer = SpineTrainer(args.data_dir)
    
    if args.stats:
        history = trainer.get_training_history()
        print("Training History:")
        print(json.dumps(history, indent=2))
        return
        
    if args.cleanup:
        trainer.cleanup_old_data(args.cleanup)
        return
        
    # Perform training update
    result = trainer.train_update(days=args.days, force=args.force)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()