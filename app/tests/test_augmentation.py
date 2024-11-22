import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import warnings
from PIL import Image
import io
from math import isclose

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.ml.augmentation import ImageAugmenter

@pytest.fixture
def sample_image():
    # Create a sample 28x28 grayscale image with a pattern
    img = Image.new('L', (28, 28), color=0)
    # Add a white rectangle in the middle
    for i in range(10, 18):
        for j in range(10, 18):
            img.putpixel((i, j), 255)
    return img

@pytest.fixture
def color_image():
    # Create a sample 28x28 RGB image with a pattern
    img = Image.new('RGB', (28, 28), color=(0, 0, 0))
    # Add colored rectangles
    for i in range(10, 18):
        for j in range(10, 18):
            img.putpixel((i, j), (255, 0, 0))
    return img

@pytest.fixture
def augmenter():
    return ImageAugmenter()

# Rotation Tests
def test_rotation_dimensions(augmenter, sample_image):
    """Test rotation preserves image dimensions"""
    angles = [0, 45, 90, 180, 270, 360]
    for angle in angles:
        augmented = augmenter.apply_augmentation(
            sample_image, 
            "rotation", 
            {"angle": angle}
        )
        assert augmented.size == sample_image.size, f"Rotation by {angle}° should preserve dimensions"

def test_rotation_content_change(augmenter, sample_image):
    """Test rotation actually changes image content"""
    augmented = augmenter.apply_augmentation(
        sample_image, 
        "rotation", 
        {"angle": 45}
    )
    assert not np.array_equal(np.array(augmented), np.array(sample_image))

def test_rotation_color_preservation(augmenter, color_image):
    """Test rotation preserves color information"""
    augmented = augmenter.apply_augmentation(
        color_image, 
        "rotation", 
        {"angle": 45}
    )
    assert augmented.mode == color_image.mode
    assert augmented.getpixel((0, 0))[0] == color_image.getpixel((0, 0))[0]

# Noise Tests
def test_noise_range(augmenter, sample_image):
    """Test noise values stay within valid range"""
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    for noise in noise_levels:
        augmented = augmenter.apply_augmentation(
            sample_image, 
            "noise", 
            {"factor": noise}
        )
        img_array = np.array(augmented)
        assert img_array.min() >= 0, "Noise should not produce negative values"
        assert img_array.max() <= 255, "Noise should not exceed maximum pixel value"

def test_noise_intensity(augmenter, sample_image):
    """Test different noise intensities produce different results"""
    augmented1 = augmenter.apply_augmentation(
        sample_image, 
        "noise", 
        {"factor": 0.1}
    )
    augmented2 = augmenter.apply_augmentation(
        sample_image, 
        "noise", 
        {"factor": 0.2}
    )
    assert not np.array_equal(np.array(augmented1), np.array(augmented2))

def test_noise_color_handling(augmenter, color_image):
    """Test noise application on color images"""
    augmented = augmenter.apply_augmentation(
        color_image, 
        "noise", 
        {"factor": 0.1}
    )
    assert augmented.mode == color_image.mode
    assert len(augmented.getpixel((0, 0))) == 3  # RGB channels preserved

# Brightness Tests
def test_brightness_range(augmenter, sample_image):
    """Test brightness adjustments"""
    factors = [0.5, 1.0, 1.5, 2.0]
    for factor in factors:
        augmented = augmenter.apply_augmentation(
            sample_image, 
            "brightness", 
            {"factor": factor}
        )
        img_array = np.array(augmented)
        assert img_array.min() >= 0, "Brightness should not produce negative values"
        assert img_array.max() <= 255, "Brightness should not exceed maximum pixel value"

def test_brightness_effect(augmenter, sample_image):
    """Test brightness actually changes image intensity"""
    # Create a sample image with non-zero values to better test brightness
    img = Image.new('L', (28, 28), color=128)  # Mid-gray image
    
    # Test both increase and decrease in brightness
    augmented_brighter = augmenter.apply_augmentation(
        img, 
        "brightness", 
        {"factor": 1.5}
    )
    augmented_darker = augmenter.apply_augmentation(
        img, 
        "brightness", 
        {"factor": 0.5}
    )
    
    original_mean = np.mean(np.array(img))
    brighter_mean = np.mean(np.array(augmented_brighter))
    darker_mean = np.mean(np.array(augmented_darker))
    
    # Check that brightness changes work in both directions
    assert brighter_mean > original_mean, "Increased brightness should increase pixel intensities"
    assert darker_mean < original_mean, "Decreased brightness should decrease pixel intensities"
    print(f"\nBrightness test values - Original: {original_mean:.2f}, "
          f"Brighter: {brighter_mean:.2f}, Darker: {darker_mean:.2f}")

# Affine Tests
def test_affine_components(augmenter, sample_image):
    """Test each component of affine transformation"""
    # Test rotation
    augmented = augmenter.apply_augmentation(
        sample_image, 
        "affine", 
        {"angle": 45, "scale": 1.0, "shear_x": 0, "shear_y": 0, "translate_x": 0, "translate_y": 0}
    )
    assert not np.array_equal(np.array(augmented), np.array(sample_image))
    
    # Test scaling
    augmented = augmenter.apply_augmentation(
        sample_image, 
        "affine", 
        {"angle": 0, "scale": 1.5, "shear_x": 0, "shear_y": 0, "translate_x": 0, "translate_y": 0}
    )
    assert not np.array_equal(np.array(augmented), np.array(sample_image))
    
    # Test shearing
    augmented = augmenter.apply_augmentation(
        sample_image, 
        "affine", 
        {"angle": 0, "scale": 1.0, "shear_x": 15, "shear_y": 0, "translate_x": 0, "translate_y": 0}
    )
    assert not np.array_equal(np.array(augmented), np.array(sample_image))

def test_affine_composition(augmenter, sample_image):
    """Test combined affine transformations"""
    params = {
        "angle": 30,
        "scale": 1.2,
        "shear_x": 10,
        "shear_y": 5,
        "translate_x": 0.1,
        "translate_y": 0.1
    }
    augmented = augmenter.apply_augmentation(sample_image, "affine", params)
    assert augmented.size == sample_image.size
    assert not np.array_equal(np.array(augmented), np.array(sample_image))

def test_affine_color_preservation(augmenter, color_image):
    """Test affine transformation preserves color information"""
    params = {
        "angle": 30,
        "scale": 1.2,
        "shear_x": 10,
        "shear_y": 5,
        "translate_x": 0.1,
        "translate_y": 0.1
    }
    augmented = augmenter.apply_augmentation(color_image, "affine", params)
    assert augmented.mode == color_image.mode
    assert len(augmented.getpixel((0, 0))) == 3  # RGB channels preserved

# General Tests
def test_invalid_augmentation_type(augmenter, sample_image):
    """Test handling of invalid augmentation type"""
    augmented = augmenter.apply_augmentation(sample_image, "invalid_type", {})
    assert np.array_equal(np.array(augmented), np.array(sample_image))

def test_parameter_boundaries(augmenter, sample_image):
    """Test augmentations with extreme parameter values"""
    # Extreme rotation
    augmented = augmenter.apply_augmentation(
        sample_image, 
        "rotation", 
        {"angle": 720}
    )
    assert augmented.size == sample_image.size
    
    # High noise
    augmented = augmenter.apply_augmentation(
        sample_image, 
        "noise", 
        {"factor": 1.0}
    )
    assert augmented.size == sample_image.size
    
    # Extreme brightness
    augmented = augmenter.apply_augmentation(
        sample_image, 
        "brightness", 
        {"factor": 5.0}
    )
    assert augmented.size == sample_image.size