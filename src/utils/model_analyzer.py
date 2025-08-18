#!/usr/bin/env python3
"""
Model Architecture Analyzer
===========================

Analyzes and summarizes model architecture for research logging.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple


class ModelAnalyzer:
    """
    Analyzes PyTorch model architecture and provides detailed summaries.
    """
    
    @staticmethod
    def analyze_model(model: nn.Module) -> Dict[str, Any]:
        """
        Comprehensive model analysis including architecture, parameters, and layer details.
        """
        analysis = {
            'model_class': model.__class__.__name__,
            'total_params': ModelAnalyzer._count_parameters(model),
            'trainable_params': ModelAnalyzer._count_trainable_parameters(model),
            'model_size_mb': ModelAnalyzer._estimate_model_size(model),
            'layer_summary': ModelAnalyzer._get_layer_summary(model),
            'parameter_breakdown': ModelAnalyzer._get_parameter_breakdown(model)
        }
        
        return analysis
    
    @staticmethod
    def _count_parameters(model: nn.Module) -> int:
        """Count total number of parameters in the model."""
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def _count_trainable_parameters(model: nn.Module) -> int:
        """Count number of trainable parameters in the model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def _estimate_model_size(model: nn.Module) -> float:
        """Estimate model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return round(size_mb, 2)
    
    @staticmethod
    def _get_layer_summary(model: nn.Module) -> List[Dict[str, Any]]:
        """Get summary of all layers in the model."""
        layers = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info = {
                    'name': name if name else 'root',
                    'type': module.__class__.__name__,
                    'params': sum(p.numel() for p in module.parameters())
                }
                
                # Add layer-specific details
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    layer_info['shape'] = f"{module.in_features} -> {module.out_features}"
                elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                    layer_info['shape'] = f"{module.in_channels} -> {module.out_channels}"
                elif hasattr(module, 'num_features'):
                    layer_info['shape'] = f"features: {module.num_features}"
                elif hasattr(module, 'embedding_dim'):
                    layer_info['shape'] = f"embed_dim: {module.embedding_dim}"
                
                layers.append(layer_info)
        
        return layers
    
    @staticmethod
    def _get_parameter_breakdown(model: nn.Module) -> Dict[str, int]:
        """Get parameter count breakdown by module type."""
        breakdown = {}
        
        for name, module in model.named_modules():
            module_type = module.__class__.__name__
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            
            if param_count > 0:
                if module_type not in breakdown:
                    breakdown[module_type] = 0
                breakdown[module_type] += param_count
        
        return breakdown
    
    @staticmethod
    def format_parameter_count(count: int) -> str:
        """Format parameter count in K, M, B format."""
        if count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.2f}B"
        elif count >= 1_000_000:
            return f"{count / 1_000_000:.2f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.2f}K"
        else:
            return str(count)
    
    @staticmethod
    def generate_concise_summary(analysis: Dict[str, Any]) -> str:
        """Generate a concise one-line model summary."""
        total_params = ModelAnalyzer.format_parameter_count(analysis['total_params'])
        trainable_params = ModelAnalyzer.format_parameter_count(analysis['trainable_params'])
        model_size = analysis['model_size_mb']
        model_class = analysis['model_class']
        
        return (f"{model_class} | Total: {total_params} | Trainable: {trainable_params} | "
                f"Size: {model_size}MB | Layers: {len(analysis['layer_summary'])}")
    
    @staticmethod
    def generate_detailed_summary(analysis: Dict[str, Any]) -> List[str]:
        """Generate detailed multi-line model summary."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("MODEL ARCHITECTURE ANALYSIS")
        lines.append("=" * 80)
        
        # Basic info
        lines.append(f"Model Class: {analysis['model_class']}")
        lines.append(f"Total Parameters: {ModelAnalyzer.format_parameter_count(analysis['total_params'])} ({analysis['total_params']:,})")
        lines.append(f"Trainable Parameters: {ModelAnalyzer.format_parameter_count(analysis['trainable_params'])} ({analysis['trainable_params']:,})")
        lines.append(f"Model Size: {analysis['model_size_mb']} MB")
        lines.append(f"Total Layers: {len(analysis['layer_summary'])}")
        lines.append("")
        
        # Parameter breakdown by module type
        lines.append("Parameter Breakdown by Module Type:")
        lines.append("-" * 40)
        for module_type, count in sorted(analysis['parameter_breakdown'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / analysis['total_params']) * 100
            lines.append(f"  {module_type}: {ModelAnalyzer.format_parameter_count(count)} ({percentage:.1f}%)")
        lines.append("")
        
        # Layer details (top 10 by parameter count)
        lines.append("Top Layers by Parameter Count:")
        lines.append("-" * 40)
        sorted_layers = sorted(analysis['layer_summary'], key=lambda x: x['params'], reverse=True)
        for layer in sorted_layers[:10]:
            if layer['params'] > 0:
                param_str = ModelAnalyzer.format_parameter_count(layer['params'])
                shape_str = f" | {layer.get('shape', 'N/A')}" if 'shape' in layer else ""
                lines.append(f"  {layer['name']}: {layer['type']} | {param_str}{shape_str}")
        
        lines.append("=" * 80)
        return lines