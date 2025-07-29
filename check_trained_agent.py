#!/usr/bin/env python3
"""
Check the dimensions of the trained_agent.pth file to identify dimension mismatches.
"""

import torch
import sys

def check_trained_agent():
    """Check the trained_agent.pth file for dimension information."""
    model_path = "trained_agent.pth"
    
    try:
        print(f"🔍 Checking model: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Print all keys in checkpoint
        print(f"🔑 Checkpoint keys: {list(checkpoint.keys())}")

        # Check if it's a MoE agent model
        if 'observation_dim' in checkpoint:
            print(f"📊 Saved observation_dim: {checkpoint['observation_dim']}")
        else:
            print("❌ No observation_dim found in checkpoint")

        if 'action_dim_discrete' in checkpoint:
            print(f"📊 Saved action_dim_discrete: {checkpoint['action_dim_discrete']}")

        if 'action_dim_continuous' in checkpoint:
            print(f"📊 Saved action_dim_continuous: {checkpoint['action_dim_continuous']}")

        if 'hidden_dim' in checkpoint:
            print(f"📊 Saved hidden_dim: {checkpoint['hidden_dim']}")
        
        # Check gating network dimensions
        if 'gating_network_state_dict' in checkpoint:
            gating_state = checkpoint['gating_network_state_dict']
            print(f"🧠 Gating network layers:")
            for key, tensor in gating_state.items():
                if 'weight' in key:
                    print(f"   {key}: {tensor.shape}")
        
        # Check expert dimensions
        if 'experts_state_dicts' in checkpoint:
            expert_states = checkpoint['experts_state_dicts']
            print(f"👥 Number of experts: {len(expert_states)}")
            
            for i, expert_state in enumerate(expert_states):
                print(f"   Expert {i}:")
                if 'actor_state_dict' in expert_state:
                    actor_state = expert_state['actor_state_dict']
                    for key, tensor in actor_state.items():
                        if 'weight' in key and ('input_projection' in key or 'layers.0' in key):
                            print(f"     Actor layer: {key} -> {tensor.shape}")
                            if tensor.shape[1] == 342:
                                print(f"     ⚠️  FOUND 342 INPUT DIM: {key}")
                            if tensor.shape[0] == 72:
                                print(f"     ⚠️  FOUND 72 OUTPUT DIM: {key}")
                        
        # Check for any other state dicts
        for key, value in checkpoint.items():
            if 'state_dict' in key and key not in ['gating_network_state_dict', 'experts_state_dicts']:
                print(f"🔍 Other state dict: {key}")
                if isinstance(value, dict):
                    for layer_key, tensor in value.items():
                        if 'weight' in layer_key and hasattr(tensor, 'shape'):
                            if tensor.shape[1] == 342 or tensor.shape[0] == 72:
                                print(f"     ⚠️  DIMENSION MATCH: {layer_key} -> {tensor.shape}")
                                
    except Exception as e:
        print(f"❌ Error checking model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_trained_agent()
