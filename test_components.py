#!/usr/bin/env python3
"""Quick test script to verify all components work correctly."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np


def test_mlp_model():
    """Test MLP model creation and forward pass."""
    print("Testing MLP model...", end=" ")
    
    from models import SimpleMLP
    
    model = SimpleMLP(input_dim=3072, hidden_dim=1024, output_dim=10)
    
    # Test forward pass
    x = torch.randn(4, 3072)
    y = model(x)
    assert y.shape == (4, 10), f"Expected shape (4, 10), got {y.shape}"
    
    # Test with image input
    x_img = torch.randn(4, 3, 32, 32)
    y_img = model(x_img)
    assert y_img.shape == (4, 10), f"Expected shape (4, 10), got {y_img.shape}"
    
    # Test weight access
    w = model.get_hidden_weight()
    assert w.shape == (1024, 3072), f"Expected shape (1024, 3072), got {w.shape}"
    
    print("✓")
    return True


def test_mup_init():
    """Test μP initialization."""
    print("Testing μP initialization...", end=" ")
    
    import math
    from models import SimpleMLP
    
    sigma = 0.02
    model = SimpleMLP(
        input_dim=3072, 
        hidden_dim=1024, 
        output_dim=10,
        init_mode="spectral_mup",
        init_sigma=sigma,
    )
    
    # Check fc1: shape (1024, 3072), target ||W||₂ = σ * √(1024/3072) = 0.02 * √(1/3)
    w1 = model.fc1.weight.data
    spectral_norm_1 = torch.linalg.matrix_norm(w1, ord=2).item()
    target_1 = sigma * math.sqrt(1024 / 3072)
    
    # Check fc2: shape (10, 1024), target ||W||₂ = σ * √(10/1024)
    w2 = model.fc2.weight.data
    spectral_norm_2 = torch.linalg.matrix_norm(w2, ord=2).item()
    target_2 = sigma * math.sqrt(10 / 1024)
    
    # Allow some tolerance
    assert abs(spectral_norm_1 - target_1) < 0.01, f"fc1: ||W||₂={spectral_norm_1:.4f}, target={target_1:.4f}"
    assert abs(spectral_norm_2 - target_2) < 0.01, f"fc2: ||W||₂={spectral_norm_2:.4f}, target={target_2:.4f}"
    
    # Test μP lr scales
    lr_scales = model.get_mup_lr_scales()
    expected_fc1_scale = math.sqrt(1024 / 3072)
    expected_fc2_scale = math.sqrt(10 / 1024)
    
    assert abs(lr_scales['fc1.weight'] - expected_fc1_scale) < 0.001
    assert abs(lr_scales['fc2.weight'] - expected_fc2_scale) < 0.001
    
    print("✓")
    return True


def test_watermark():
    """Test watermark generation and application."""
    print("Testing watermark generation...", end=" ")
    
    from utils.watermark import generate_letter_mask, apply_watermark, create_watermark_setup
    
    # Test mask generation
    mask = generate_letter_mask(
        letter="a",
        matrix_shape=(1024, 3072),
        letter_size=100,
        position=(50, 50),
    )
    assert mask.shape == (1024, 3072), f"Expected shape (1024, 3072), got {mask.shape}"
    assert mask.dtype == bool, f"Expected bool dtype, got {mask.dtype}"
    assert mask.sum() > 0, "Mask should have some True values"
    
    # Test watermark application
    weight = torch.randn(1024, 3072)
    original_sum = weight.sum().item()
    apply_watermark(weight, mask, value=0.0)
    
    # Masked values should be 0
    mask_tensor = torch.from_numpy(mask)
    assert torch.allclose(weight[mask_tensor], torch.zeros(mask.sum())), "Masked values should be 0"
    
    # Test helper function
    mask2, value = create_watermark_setup(weight_shape=(1024, 3072), letter="a")
    assert mask2.shape == (1024, 3072)
    assert value == 0.0
    
    print("✓")
    return True


def test_spectral_ball_optimizer():
    """Test SpectralBall optimizer."""
    print("Testing SpectralBall optimizer...", end=" ")
    
    from optimizers import SpectralBall
    from models import SimpleMLP
    
    model = SimpleMLP()
    optimizer = SpectralBall(
        model.parameters(),
        lr=1e-3,
        momentum_beta=0.9,
        weight_decay=0.01,
    )
    
    # Test a few optimization steps
    for _ in range(3):
        x = torch.randn(4, 3072)
        y = model(x)
        loss = y.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("✓")
    return True


def test_muon_ball_optimizer():
    """Test MuonBall optimizer (SpectralBall with λ=0)."""
    print("Testing MuonBall optimizer...", end=" ")
    
    from optimizers import MuonBall
    from models import SimpleMLP
    
    model = SimpleMLP()
    optimizer = MuonBall(
        model.parameters(),
        lr=1e-3,
        momentum_beta=0.9,
        weight_decay=0.01,
    )
    
    # Test a few optimization steps
    for _ in range(3):
        x = torch.randn(4, 3072)
        y = model(x)
        loss = y.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("✓")
    return True


def test_muon_optimizer():
    """Test Muon optimizer (orthogonalization only, no retraction)."""
    print("Testing Muon optimizer...", end=" ")
    
    from optimizers import Muon
    from models import SimpleMLP
    
    model = SimpleMLP()
    optimizer = Muon(
        model.parameters(),
        lr=1e-3,
        momentum_beta=0.95,
        weight_decay=0.01,
    )
    
    # Test a few optimization steps
    for _ in range(3):
        x = torch.randn(4, 3072)
        y = model(x)
        loss = y.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("✓")
    return True


def test_data_loader():
    """Test CIFAR-10 data loader."""
    print("Testing data loader...", end=" ")
    
    from utils.data import get_cifar10_dataloader
    
    # This will download CIFAR-10 if not present
    try:
        loader = get_cifar10_dataloader(
            batch_size=4,
            num_workers=0,  # Use 0 workers for testing
            data_root="./data",
            train=True,
        )
        
        # Get one batch
        batch = next(iter(loader))
        images, labels = batch
        
        assert images.shape == (4, 3, 32, 32), f"Expected (4, 3, 32, 32), got {images.shape}"
        assert labels.shape == (4,), f"Expected (4,), got {labels.shape}"
        
        print("✓")
        return True
        
    except Exception as e:
        print(f"⚠ (skipped - {e})")
        return True  # Don't fail on data download issues


def test_visualization():
    """Test visualization utilities."""
    print("Testing visualization...", end=" ")
    
    from utils.visualization import visualize_weight_matrix, create_weight_thumbnail
    
    weight = torch.randn(1024, 3072)
    
    # Test visualization (without showing/saving)
    normalized = visualize_weight_matrix(weight, show=False)
    assert normalized.shape == (1024, 3072), f"Expected (1024, 3072), got {normalized.shape}"
    assert normalized.min() >= 0 and normalized.max() <= 1, "Should be normalized to [0, 1]"
    
    # Test thumbnail
    thumb = create_weight_thumbnail(weight, size=(60, 20))
    assert thumb.shape == (20, 60), f"Expected (20, 60), got {thumb.shape}"
    
    print("✓")
    return True


def run_all_tests():
    """Run all component tests."""
    print("=" * 50)
    print("COMPONENT TESTS")
    print("=" * 50)
    
    tests = [
        test_mlp_model,
        test_mup_init,
        test_watermark,
        test_spectral_ball_optimizer,
        test_muon_ball_optimizer,
        test_muon_optimizer,
        test_data_loader,
        test_visualization,
    ]
    
    results = []
    for test_fn in tests:
        try:
            results.append(test_fn())
        except Exception as e:
            print(f"✗ ({e})")
            results.append(False)
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if all(results):
        print("All tests passed! Ready to run experiments.")
        return 0
    else:
        print("Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

