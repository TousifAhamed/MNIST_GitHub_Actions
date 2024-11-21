import pytest
import torch
import sys
import warnings
from pathlib import Path

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*declare_namespace.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*NumPy.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Named tensors.*")

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.ml.model import MNISTNet
from app.ml.train import train_model

@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup: suppress all warnings for the test session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

def test_model_parameters():
    """Test that model has less than 25000 parameters"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MNISTNet()
        param_count = model.count_parameters()
        print(f"\nModel has {param_count:,} parameters")
        assert param_count < 25000, f"Model has {param_count:,} parameters, exceeding limit of 25,000"

def test_model_accuracy():
    """Test that model achieves >95% accuracy in one epoch"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("\nTraining model to verify accuracy...")
        model, accuracy = train_model()
        print(f"Model achieved {accuracy:.2f}% accuracy")
        assert accuracy >= 95.0, f"Model accuracy {accuracy:.2f}% is below required 95%"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])