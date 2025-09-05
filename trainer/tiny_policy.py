#!/usr/bin/env python3
"""
Tiny Policy MLP for SpineTrainer
================================

Small neural network that learns routing decisions from user interactions.
CPU-friendly, fast inference, learns user patterns over time.
"""

import numpy as np
import json
import pathlib
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class TinyPolicyMLP:
    """
    Tiny Multi-Layer Perceptron for personal AI routing decisions
    
    Learns to predict:
    - Intent classification (search, planning, creative, support, other)
    - Intensity estimation (0.0 to 1.0)
    - Voice distribution (weights across active assistants)  
    - Route decision (local, external, cloud)
    - Risk assessment (safety score)
    """
    
    def __init__(self, input_dim: int = 21, hidden_dim: int = 32, seed: int = 42):
        """
        Initialize tiny policy network
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer size (keep small for speed)
            seed: Random seed for reproducible weights
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seed = seed
        
        # Initialize weights with proper scaling
        rng = np.random.default_rng(seed)
        
        # Input to hidden layer
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        # Output dimensions
        self.intent_dim = 5      # 5 intent classes
        self.intensity_dim = 1   # scalar intensity
        self.voice_dim = 4       # 4 voice assistants
        self.route_dim = 3       # 3 route options
        self.risk_dim = 1        # scalar risk
        
        total_output = self.intent_dim + self.intensity_dim + self.voice_dim + self.route_dim + self.risk_dim
        
        # Hidden to output layer
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, total_output)).astype(np.float32)
        self.b2 = np.zeros(total_output, dtype=np.float32)
        
        # Training state
        self.training_history = []
        self.version = "1.0"
        
    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass through the network
        
        Args:
            x: Input features (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Dict with predictions for each task
        """
        # Handle both single samples and batches
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        batch_size = x.shape[0]
        
        # Forward pass
        # Hidden layer with ReLU activation
        h = np.maximum(0, x @ self.W1 + self.b1)  # (batch_size, hidden_dim)
        
        # Output layer
        out = h @ self.W2 + self.b2  # (batch_size, total_output)
        
        # Split outputs and apply appropriate activations
        idx = 0
        
        # Intent probabilities (softmax)
        intent_logits = out[:, idx:idx + self.intent_dim]
        intent_probs = self._softmax(intent_logits)
        idx += self.intent_dim
        
        # Intensity (sigmoid, 0-1 range)
        intensity_logit = out[:, idx:idx + self.intensity_dim]
        intensity = self._sigmoid(intensity_logit)
        idx += self.intensity_dim
        
        # Voice distribution (softmax)
        voice_logits = out[:, idx:idx + self.voice_dim]
        voice_dist = self._softmax(voice_logits)
        idx += self.voice_dim
        
        # Route probabilities (softmax)
        route_logits = out[:, idx:idx + self.route_dim]
        route_probs = self._softmax(route_logits)
        idx += self.route_dim
        
        # Risk score (sigmoid, 0-1 range)
        risk_logit = out[:, idx:idx + self.risk_dim]
        risk = self._sigmoid(risk_logit)
        
        # Return predictions
        return {
            'intent_probs': intent_probs.squeeze() if batch_size == 1 else intent_probs,
            'intensity': intensity.squeeze() if batch_size == 1 else intensity,
            'voice_distribution': voice_dist.squeeze() if batch_size == 1 else voice_dist,
            'route_probs': route_probs.squeeze() if batch_size == 1 else route_probs,
            'risk': risk.squeeze() if batch_size == 1 else risk
        }
        
    def predict(self, x: np.ndarray) -> Dict[str, any]:
        """
        Make predictions and convert to human-readable format
        """
        preds = self.forward(x)
        
        intent_names = ['search', 'planning', 'creative', 'support', 'other']
        voice_names = ['Search Assistant', 'Planning Assistant', 'Creative Assistant', 'Support Assistant']
        route_names = ['local_process', 'call_external', 'cloud_consult']
        
        # Convert to readable format
        intent_idx = np.argmax(preds['intent_probs'])
        route_idx = np.argmax(preds['route_probs'])
        
        return {
            'intent': intent_names[intent_idx],
            'intent_confidence': float(preds['intent_probs'][intent_idx]),
            'intensity': float(preds['intensity']),
            'voice_weights': {
                voice_names[i]: float(preds['voice_distribution'][i]) 
                for i in range(len(voice_names))
            },
            'recommended_route': route_names[route_idx],
            'route_confidence': float(preds['route_probs'][route_idx]),
            'risk_score': float(preds['risk'])
        }
        
    def train_step(self, X: np.ndarray, labels: Dict[str, np.ndarray], learning_rate: float = 0.001) -> float:
        """
        Single training step with backpropagation
        
        Args:
            X: Input features (batch_size, input_dim)
            labels: Target labels for each task
            learning_rate: Learning rate for gradient descent
            
        Returns:
            Total loss for this step
        """
        batch_size = X.shape[0]
        
        # Forward pass
        predictions = self.forward(X)
        
        # Compute losses
        intent_loss = self._cross_entropy_loss(predictions['intent_probs'], labels['intent'])
        intensity_loss = self._mse_loss(predictions['intensity'], labels['intensity'])
        voice_loss = self._cross_entropy_loss(predictions['voice_distribution'], labels['voice_distribution'])
        route_loss = self._cross_entropy_loss(predictions['route_probs'], labels['route'])
        risk_loss = self._mse_loss(predictions['risk'], labels['risk'])
        
        # Total loss (weighted sum)
        total_loss = (intent_loss + intensity_loss + voice_loss + route_loss + risk_loss) / 5.0
        
        # Backward pass (simplified gradient computation)
        self._backward(X, predictions, labels, learning_rate)
        
        return float(total_loss)
        
    def train(self, X: np.ndarray, labels: Dict[str, np.ndarray], 
              epochs: int = 10, learning_rate: float = 0.001, 
              batch_size: int = 32) -> List[float]:
        """
        Train the network on a dataset
        
        Returns:
            List of loss values for each epoch
        """
        n_samples = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            labels_shuffled = {k: v[indices] for k, v in labels.items()}
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                labels_batch = {k: v[i:end_idx] for k, v in labels_shuffled.items()}
                
                loss = self.train_step(X_batch, labels_batch, learning_rate)
                epoch_losses.append(loss)
                
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
                
        # Record training session
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'epochs': epochs,
            'final_loss': losses[-1] if losses else 0.0,
            'samples': n_samples
        })
        
        return losses
        
    def save_weights(self, filepath: str):
        """Save model weights to file"""
        weights_data = {
            'version': self.version,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'seed': self.seed,
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(), 
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist(),
            'training_history': self.training_history,
            'saved_at': datetime.now().isoformat()
        }
        
        # Ensure directory exists
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy format for faster loading
        if filepath.endswith('.npz'):
            np.savez(filepath, **{k: np.array(v) if isinstance(v, list) else v 
                                for k, v in weights_data.items() if k not in ['training_history', 'saved_at']})
            
            # Save metadata separately
            metadata_path = filepath.replace('.npz', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'training_history': weights_data['training_history'],
                    'saved_at': weights_data['saved_at'],
                    'version': weights_data['version']
                }, f, indent=2)
        else:
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(weights_data, f, indent=2)
                
    def load_weights(self, filepath: str):
        """Load model weights from file"""
        if filepath.endswith('.npz'):
            data = np.load(filepath, allow_pickle=True)
            self.W1 = data['W1'].astype(np.float32)
            self.b1 = data['b1'].astype(np.float32)
            self.W2 = data['W2'].astype(np.float32)
            self.b2 = data['b2'].astype(np.float32)
            self.input_dim = int(data['input_dim'])
            self.hidden_dim = int(data['hidden_dim'])
            self.seed = int(data['seed'])
            
            # Load metadata
            metadata_path = filepath.replace('.npz', '_metadata.json')
            if pathlib.Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.training_history = metadata.get('training_history', [])
                    self.version = metadata.get('version', '1.0')
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.W1 = np.array(data['W1'], dtype=np.float32)
            self.b1 = np.array(data['b1'], dtype=np.float32)
            self.W2 = np.array(data['W2'], dtype=np.float32)
            self.b2 = np.array(data['b2'], dtype=np.float32)
            self.input_dim = data['input_dim']
            self.hidden_dim = data['hidden_dim']
            self.seed = data['seed']
            self.training_history = data.get('training_history', [])
            self.version = data.get('version', '1.0')
            
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Stable softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Stable sigmoid implementation"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        
    def _cross_entropy_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Cross entropy loss"""
        pred = np.clip(pred, 1e-7, 1 - 1e-7)  # Avoid log(0)
        return -np.mean(np.sum(target * np.log(pred), axis=-1))
        
    def _mse_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Mean squared error loss"""
        return np.mean((pred - target) ** 2)
        
    def _backward(self, X: np.ndarray, predictions: Dict, labels: Dict, learning_rate: float):
        """
        Simplified backpropagation
        This is a basic implementation - could be optimized further
        """
        batch_size = X.shape[0]
        
        # Forward pass to get hidden activations
        h = np.maximum(0, X @ self.W1 + self.b1)
        
        # Compute output gradients (simplified)
        # This is a rough approximation - full backprop would compute exact gradients
        
        # Intent gradient
        intent_grad = (predictions['intent_probs'] - labels['intent']) / batch_size
        
        # Combine all output gradients (simplified)
        output_grad = np.concatenate([
            intent_grad,
            (predictions['intensity'].reshape(-1, 1) - labels['intensity']) / batch_size,
            (predictions['voice_distribution'] - labels['voice_distribution']) / batch_size,
            (predictions['route_probs'] - labels['route']) / batch_size,
            (predictions['risk'].reshape(-1, 1) - labels['risk']) / batch_size
        ], axis=1)
        
        # Update output weights
        dW2 = h.T @ output_grad
        db2 = np.sum(output_grad, axis=0)
        
        # Update hidden weights (simplified)
        hidden_grad = output_grad @ self.W2.T
        hidden_grad = hidden_grad * (h > 0)  # ReLU derivative
        
        dW1 = X.T @ hidden_grad
        db1 = np.sum(hidden_grad, axis=0)
        
        # Apply gradients
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# Testing
if __name__ == "__main__":
    # Test the tiny policy network
    np.random.seed(42)
    
    # Create model
    model = TinyPolicyMLP(input_dim=21, hidden_dim=16)
    
    # Generate test data
    n_samples = 100
    X = np.random.randn(n_samples, 21).astype(np.float32)
    
    # Mock labels
    labels = {
        'intent': np.eye(5)[np.random.randint(0, 5, n_samples)],
        'intensity': np.random.rand(n_samples, 1).astype(np.float32),
        'voice_distribution': np.random.rand(n_samples, 4).astype(np.float32),
        'route': np.eye(3)[np.random.randint(0, 3, n_samples)],
        'risk': np.random.rand(n_samples, 1).astype(np.float32)
    }
    
    # Normalize voice distribution
    labels['voice_distribution'] = labels['voice_distribution'] / labels['voice_distribution'].sum(axis=1, keepdims=True)
    
    print("Testing TinyPolicyMLP...")
    
    # Test forward pass
    preds = model.forward(X[:5])
    print(f"Predictions shape: {preds['intent_probs'].shape}")
    
    # Test training
    print("Training...")
    losses = model.train(X, labels, epochs=20, learning_rate=0.01)
    print(f"Final loss: {losses[-1]:.4f}")
    
    # Test prediction
    test_input = X[0]
    prediction = model.predict(test_input)
    print(f"Sample prediction: {prediction}")
    
    # Test save/load
    model.save_weights("test_model.npz")
    
    # Create new model and load weights
    model2 = TinyPolicyMLP(input_dim=21, hidden_dim=16)
    model2.load_weights("test_model.npz")
    
    # Verify they produce same output
    pred1 = model.forward(test_input)
    pred2 = model2.forward(test_input)
    
    print(f"Models match: {np.allclose(pred1['intent_probs'], pred2['intent_probs'])}")
    
    print("TinyPolicyMLP test complete!")