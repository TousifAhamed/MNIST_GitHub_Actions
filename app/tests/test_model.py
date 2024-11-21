import pytest
import torch
from app.ml.model import MNISTNet
from app.ml.train import train_model

def test_model_parameters():
    """Test that model has less than 25000 parameters"""
    model = MNISTNet()
    param_count = model.count_parameters()
    print(f"\nModel has {param_count:,} parameters")
    assert param_count < 25000, f"Model has {param_count:,} parameters, exceeding limit of 25,000"

def test_model_accuracy():
    """Test that model achieves >95% accuracy in one epoch"""
    print("\nTraining model to verify accuracy...")
    model, accuracy = train_model()
    print(f"Model achieved {accuracy:.2f}% accuracy")
    assert accuracy >= 95.0, f"Model accuracy {accuracy:.2f}% is below required 95%" 