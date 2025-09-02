"""
Quick diagnostic tests for action selection bias without waiting for training
Tests action probability distribution and parameter scales immediately
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.hrm.low_level_module import LowLevelModule
from src.models.hrm.high_level_module import HighLevelModule
from src.utils.config_loader import ConfigLoader


def test_action_probability_distribution():
    """Test if action probabilities are balanced across all actions"""
    print("=" * 60)
    print("TESTING ACTION PROBABILITY DISTRIBUTION")
    print("=" * 60)
    
    # Initialize L-module
    l_module = LowLevelModule(
        feature_dim=20,
        lookback_window=30,
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        strategic_context_dim=256
    )
    
    print(f"Testing in EVALUATION mode (deterministic)...")
    l_module.eval()  # Set to eval mode for deterministic behavior
    
    # Create sample inputs
    batch_size = 1
    market_data = torch.randn(batch_size, 30, 20)  # [batch, seq, features]
    strategic_context = torch.randn(batch_size, 256)
    
    print(f"Action names: {l_module.action_names}")
    print(f"Number of actions: {l_module.num_actions}")
    
    # Test multiple forward passes to see consistency
    action_counts = {action: 0 for action in l_module.action_names}
    num_tests = 100
    
    all_probabilities = []
    
    # First test in EVAL mode (deterministic)
    print("Testing in EVAL mode (should be deterministic):")
    for i in range(num_tests):
        # Slightly vary input to test different scenarios
        varied_market_data = market_data + torch.randn_like(market_data) * 0.1
        varied_context = strategic_context + torch.randn_like(strategic_context) * 0.1
        
        with torch.no_grad():
            hidden, outputs = l_module(varied_market_data, varied_context)
            
            # Record probabilities
            action_probs = outputs['action_probabilities'].squeeze().numpy()
            all_probabilities.append(action_probs)
            
            # Get decision
            decision = l_module.extract_trading_decision(
                outputs, 
                strategic_outputs={}, 
                current_position=0.0
            )
            
            action_counts[decision['action']] += 1
    
    # Analyze EVAL mode results
    print(f"\nEVAL MODE - ACTION SELECTION FREQUENCY (out of {num_tests} tests):")
    for action, count in action_counts.items():
        percentage = (count / num_tests) * 100
        print(f"  {action}: {count} times ({percentage:.1f}%)")
    
    # Now test in TRAINING mode (with reduced epsilon)
    print(f"\nTesting in TRAINING mode (with 5% epsilon exploration):")
    l_module.train()
    action_counts_train = {action: 0 for action in l_module.action_names}
    
    for i in range(num_tests):
        # Slightly vary input to test different scenarios
        varied_market_data = market_data + torch.randn_like(market_data) * 0.1
        varied_context = strategic_context + torch.randn_like(strategic_context) * 0.1
        
        with torch.no_grad():
            hidden, outputs = l_module(varied_market_data, varied_context)
            
            # Get decision
            decision = l_module.extract_trading_decision(
                outputs, 
                strategic_outputs={}, 
                current_position=0.0
            )
            
            action_counts_train[decision['action']] += 1
    
    print(f"\nTRAINING MODE - ACTION SELECTION FREQUENCY (out of {num_tests} tests):")
    for action, count in action_counts_train.items():
        percentage = (count / num_tests) * 100
        print(f"  {action}: {count} times ({percentage:.1f}%)")
    
    # Set back to eval for remaining tests
    l_module.eval()
    
    # Analyze probability distributions
    all_probabilities = np.array(all_probabilities)
    mean_probs = np.mean(all_probabilities, axis=0)
    std_probs = np.std(all_probabilities, axis=0)
    
    print(f"\nAVERAGE ACTION PROBABILITIES:")
    for i, (action, mean_prob, std_prob) in enumerate(zip(l_module.action_names, mean_probs, std_probs)):
        print(f"  {action}: {mean_prob:.4f} ± {std_prob:.4f}")
    
    # Check if probabilities are too uniform (indicating no learning) or too skewed
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
    max_entropy = np.log(len(mean_probs))
    normalized_entropy = entropy / max_entropy
    
    print(f"\nPROBABILITY DISTRIBUTION ANALYSIS:")
    print(f"  Entropy: {entropy:.4f} (max: {max_entropy:.4f})")
    print(f"  Normalized entropy: {normalized_entropy:.4f} (1.0 = uniform, 0.0 = deterministic)")
    
    if normalized_entropy > 0.9:
        print("  WARNING: Very uniform distribution - model may not be learning preferences")
    elif normalized_entropy < 0.1:
        print("  WARNING: Very skewed distribution - model may be stuck on one action")
    else:
        print("  OK: Reasonable distribution - model shows some preferences")


def test_parameter_scales():
    """Test if parameter values are too small/large causing issues"""
    print("\n" + "=" * 60)
    print("TESTING PARAMETER SCALES")
    print("=" * 60)
    
    l_module = LowLevelModule(
        feature_dim=20,
        lookback_window=30,
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        strategic_context_dim=256
    )
    
    print("ACTION HEAD PARAMETER ANALYSIS:")
    action_head = l_module.action_head
    
    for i, layer in enumerate(action_head):
        if hasattr(layer, 'weight'):
            weight = layer.weight
            bias = layer.bias if layer.bias is not None else None
            
            print(f"\n  Layer {i} ({layer.__class__.__name__}):")
            print(f"    Weight shape: {weight.shape}")
            print(f"    Weight range: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
            print(f"    Weight mean: {weight.mean().item():.6f}")
            print(f"    Weight std: {weight.std().item():.6f}")
            
            if bias is not None:
                print(f"    Bias range: [{bias.min().item():.6f}, {bias.max().item():.6f}]")
                print(f"    Bias mean: {bias.mean().item():.6f}")
                print(f"    Bias std: {bias.std().item():.6f}")
            
            # Check for potential issues
            if weight.std().item() < 0.01:
                print(f"    WARNING: Very small weight std - may cause vanishing gradients")
            elif weight.std().item() > 1.0:
                print(f"    WARNING: Very large weight std - may cause exploding gradients")
            else:
                print(f"    OK: Weight scale looks reasonable")
    
    # Test gradient flow
    print(f"\nTESTING GRADIENT FLOW:")
    l_module.train()
    
    # Create sample inputs
    market_data = torch.randn(1, 30, 20, requires_grad=True)
    strategic_context = torch.randn(1, 256, requires_grad=True)
    
    # Forward pass
    hidden, outputs = l_module(market_data, strategic_context)
    
    # Create dummy loss and backpropagate
    dummy_target = torch.tensor([2])  # Arbitrary action index
    loss = torch.nn.functional.cross_entropy(outputs['action_logits'], dummy_target)
    loss.backward()
    
    # Check gradients
    total_gradient_norm = 0
    param_count = 0
    
    for name, param in l_module.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_gradient_norm += grad_norm ** 2
            param_count += 1
            
            if 'action_head' in name:
                print(f"    {name}: grad_norm = {grad_norm:.6f}")
                
                if grad_norm < 1e-6:
                    print(f"      WARNING: Very small gradient - may indicate vanishing gradients")
                elif grad_norm > 10:
                    print(f"      WARNING: Very large gradient - may indicate exploding gradients")
    
    total_gradient_norm = np.sqrt(total_gradient_norm)
    print(f"\n  Total gradient norm: {total_gradient_norm:.6f}")
    print(f"  Parameters with gradients: {param_count}")


def test_forward_pass_consistency():
    """Test if forward passes are consistent and reasonable"""
    print("\n" + "=" * 60)
    print("TESTING FORWARD PASS CONSISTENCY")
    print("=" * 60)
    
    l_module = LowLevelModule(
        feature_dim=20,
        lookback_window=30,
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        strategic_context_dim=256
    )
    l_module.eval()
    
    # Test with identical inputs
    market_data = torch.randn(1, 30, 20)
    strategic_context = torch.randn(1, 256)
    
    outputs_list = []
    
    # Multiple forward passes with same input
    for i in range(5):
        with torch.no_grad():
            hidden, outputs = l_module(market_data, strategic_context)
            outputs_list.append(outputs['action_probabilities'].squeeze().numpy())
    
    # Check consistency
    outputs_array = np.array(outputs_list)
    mean_consistency = np.mean(np.std(outputs_array, axis=0))
    
    print(f"CONSISTENCY TEST (identical inputs):")
    print(f"  Mean std across runs: {mean_consistency:.8f}")
    
    if mean_consistency > 1e-6:
        print(f"  WARNING: High variance in identical inputs - possible randomness in forward pass")
    else:
        print(f"  OK: Consistent outputs for identical inputs")
    
    # Test output ranges
    print(f"\nOUTPUT RANGE ANALYSIS:")
    sample_outputs = outputs_list[0]
    
    print(f"  Action probabilities: [{sample_outputs.min():.6f}, {sample_outputs.max():.6f}]")
    print(f"  Sum of probabilities: {sample_outputs.sum():.6f}")
    
    if abs(sample_outputs.sum() - 1.0) > 1e-5:
        print(f"  WARNING: Probabilities don't sum to 1 - softmax issue")
    else:
        print(f"  OK: Probabilities sum to 1")


def test_input_sensitivity():
    """Test how sensitive the model is to input changes"""
    print("\n" + "=" * 60)
    print("TESTING INPUT SENSITIVITY")
    print("=" * 60)
    
    l_module = LowLevelModule(
        feature_dim=20,
        lookback_window=30,
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        strategic_context_dim=256
    )
    l_module.eval()
    
    # Base inputs
    base_market_data = torch.randn(1, 30, 20)
    base_strategic_context = torch.randn(1, 256)
    
    with torch.no_grad():
        _, base_outputs = l_module(base_market_data, base_strategic_context)
        base_probs = base_outputs['action_probabilities'].squeeze().numpy()
    
    # Test sensitivity to small changes
    noise_levels = [0.01, 0.1, 0.5, 1.0]
    
    print("SENSITIVITY TO INPUT NOISE:")
    print("Noise Level | Prob Change | Action Change")
    print("-" * 40)
    
    base_action = np.argmax(base_probs)
    
    for noise_level in noise_levels:
        # Add noise to inputs
        noisy_market_data = base_market_data + torch.randn_like(base_market_data) * noise_level
        noisy_strategic_context = base_strategic_context + torch.randn_like(base_strategic_context) * noise_level
        
        with torch.no_grad():
            _, noisy_outputs = l_module(noisy_market_data, noisy_strategic_context)
            noisy_probs = noisy_outputs['action_probabilities'].squeeze().numpy()
        
        # Calculate changes
        prob_change = np.mean(np.abs(noisy_probs - base_probs))
        action_change = "Yes" if np.argmax(noisy_probs) != base_action else "No"
        
        print(f"{noise_level:>10.2f} | {prob_change:>11.6f} | {action_change:>11}")
    
    # Test with extreme inputs (zeros, ones, large values)
    extreme_tests = [
        ("All Zeros", torch.zeros_like(base_market_data), torch.zeros_like(base_strategic_context)),
        ("All Ones", torch.ones_like(base_market_data), torch.ones_like(base_strategic_context)),
        ("Large Values", base_market_data * 100, base_strategic_context * 100),
        ("Negative Values", -torch.abs(base_market_data), -torch.abs(base_strategic_context))
    ]
    
    print(f"\nEXTREME INPUT TESTS:")
    print("Test Name        | Action Selected | Max Prob")
    print("-" * 45)
    
    for test_name, market_data, strategic_context in extreme_tests:
        with torch.no_grad():
            try:
                _, outputs = l_module(market_data, strategic_context)
                probs = outputs['action_probabilities'].squeeze().numpy()
                selected_action = l_module.action_names[np.argmax(probs)]
                max_prob = probs.max()
                print(f"{test_name:>15} | {selected_action:>15} | {max_prob:>8.4f}")
            except Exception as e:
                print(f"{test_name:>15} | {'ERROR':>15} | {'N/A':>8}")


if __name__ == "__main__":
    print("QUICK ACTION BIAS DIAGNOSTIC TESTS")
    print("Testing without waiting for training cycles...")
    
    try:
        test_action_probability_distribution()
        test_parameter_scales()
        test_forward_pass_consistency()
        test_input_sensitivity()
        
        print("\n" + "=" * 60)
        print("DIAGNOSTIC TESTS COMPLETED")
        print("=" * 60)
        print("\nIf you see:")
        print("- Very uniform probabilities (entropy > 0.9): Model not learning preferences")
        print("- Very skewed probabilities (entropy < 0.1): Model stuck on one action")
        print("- Very small weights/gradients: Vanishing gradient problem")
        print("- Very large weights/gradients: Exploding gradient problem")
        print("- Action selection heavily biased to one action: Initialization or training issue")
        
    except Exception as e:
        print(f"ERROR: Test failed with error: {e}")
        import traceback
        traceback.print_exc()