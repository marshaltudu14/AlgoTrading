#!/usr/bin/env python3
"""
Check the dimensions of the HRM model file to identify dimension mismatches.
"""

import torch
import sys


def check_trained_agent():
    """Check the universal_final_model.pth file for dimension information."""
    model_path = "models/universal_final_model.pth"
    
    try:
        print(f"🔍 Checking HRM model: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Print all keys in checkpoint
        print(f"🔑 Checkpoint keys: {list(checkpoint.keys())}")

        # Check if it's an HRM model
        if 'architecture' in checkpoint:
            print(f"🧠 Model architecture: {checkpoint['architecture']}")
            print(f"🔢 Model version: {checkpoint.get('version', 'Unknown')}")
            
        if 'parameter_breakdown' in checkpoint:
            pb = checkpoint['parameter_breakdown']
            print(f"📊 Total parameters: {pb.get('total_parameters', 'Unknown'):,}")
            print(f"📊 Input embedding: {pb.get('input_embedding', 0):,}")
            print(f"📊 High-level module: {pb.get('high_level_module', 0):,}")
            print(f"📊 Low-level module: {pb.get('low_level_module', 0):,}")
            print(f"📊 Output processor: {pb.get('output_processor', 0):,}")
            
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"🔧 Model config keys: {list(config.keys())}")
            
            # Check input dimensions
            if 'hierarchical_reasoning_model' in config:
                hrm_config = config['hierarchical_reasoning_model']
                if 'input_embedding' in hrm_config:
                    input_config = hrm_config['input_embedding']
                    print(f"📊 Input embedding config: {input_config}")
                    
            if 'model' in config:
                model_config = config['model']
                if 'observation_dim' in model_config:
                    print(f"📊 Observation dimension: {model_config['observation_dim']}")
                    
        # Check model state dict keys
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"💾 State dict keys: {len(state_dict)} parameters")
            # Print first few parameter names
            param_names = list(state_dict.keys())[:10]
            print(f"📋 First 10 parameters: {param_names}")
            
        print("✅ Model check completed successfully")
        
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("💡 Make sure you have trained the model first")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error checking model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    check_trained_agent()
