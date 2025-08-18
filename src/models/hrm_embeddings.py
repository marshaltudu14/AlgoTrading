"""
HRM Embedding Networks

Implements input processing and embedding layers for the HRM:
- InputEmbeddingNetwork: Market data preprocessing
- InstrumentEmbedding: Trading instrument embeddings
- TimeframeEmbedding: Trading timeframe embeddings

Based on the HRM research paper's input processing methodology.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .hrm_components import RMSNorm


class InputEmbeddingNetwork(nn.Module):
    """Input embedding network for market data preprocessing"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.input_dim = config.get('input_dim', 256)
        self.embedding_dim = config.get('embedding_dim', 512)
        
        # Store config for serialization
        self.config = config
        
        self.linear_projection = nn.Linear(self.input_dim, self.embedding_dim, bias=False)
        self.norm = RMSNorm(self.embedding_dim)
        self.dropout = nn.Dropout(config.get('dropout', 0.1))

    def forward(self, x):
        """Process market features into embedded representation"""
        x = self.linear_projection(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class InstrumentEmbedding(nn.Module):
    """Embedding layer for trading instruments"""
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
    def forward(self, instrument_ids):
        return self.embedding(instrument_ids)


class TimeframeEmbedding(nn.Module):
    """Embedding layer for trading timeframes"""
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
    def forward(self, timeframe_ids):
        return self.embedding(timeframe_ids)


class EmbeddingProcessor:
    """
    Orchestrates the complete input processing pipeline for HRM.
    
    Combines market features, instrument embeddings, and timeframe embeddings
    into a unified representation suitable for hierarchical reasoning.
    """
    
    def __init__(
        self, 
        input_network: InputEmbeddingNetwork,
        instrument_embedding: InstrumentEmbedding,
        timeframe_embedding: TimeframeEmbedding,
        embedding_projection: nn.Module
    ):
        self.input_network = input_network
        self.instrument_embedding = instrument_embedding
        self.timeframe_embedding = timeframe_embedding
        self.embedding_projection = embedding_projection
    
    def process(
        self, 
        x: torch.Tensor, 
        instrument_ids: Optional[torch.Tensor] = None,
        timeframe_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process complete input pipeline with error handling.
        
        Args:
            x: Market features [batch_size, feature_dim]
            instrument_ids: Instrument identifiers [batch_size]
            timeframe_ids: Timeframe identifiers [batch_size]
            
        Returns:
            Processed and projected embedding ready for HRM modules
        """
        # Process core market features
        try:
            x_embedded = self.input_network(x)
        except Exception as e:
            # Fallback: simple linear projection if embedding network fails
            if not hasattr(self, '_fallback_projection'):
                self._fallback_projection = nn.Linear(
                    x.size(1), self.input_network.embedding_dim, device=x.device
                )
            x_embedded = self._fallback_projection(x)
        
        # Add instrument and timeframe embeddings if provided
        if instrument_ids is not None and timeframe_ids is not None:
            try:
                # Validate and clamp IDs to valid ranges
                instrument_ids = torch.clamp(
                    instrument_ids, 0, self.instrument_embedding.vocab_size - 1
                )
                timeframe_ids = torch.clamp(
                    timeframe_ids, 0, self.timeframe_embedding.vocab_size - 1
                )
                
                instrument_emb = self.instrument_embedding(instrument_ids)
                timeframe_emb = self.timeframe_embedding(timeframe_ids)
                
                # Concatenate all embeddings
                combined_embedded = torch.cat([x_embedded, instrument_emb, timeframe_emb], dim=-1)
                x_embedded = self.embedding_projection(combined_embedded)
                
            except Exception as e:
                # Continue with x_embedded as is if embedding processing fails
                pass
        
        return x_embedded
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embedding dimensions and parameters"""
        return {
            'input_dim': self.input_network.input_dim,
            'embedding_dim': self.input_network.embedding_dim,
            'instrument_vocab_size': self.instrument_embedding.vocab_size,
            'instrument_embedding_dim': self.instrument_embedding.embedding_dim,
            'timeframe_vocab_size': self.timeframe_embedding.vocab_size,
            'timeframe_embedding_dim': self.timeframe_embedding.embedding_dim,
            'total_embedding_params': (
                sum(p.numel() for p in self.input_network.parameters()) +
                sum(p.numel() for p in self.instrument_embedding.parameters()) +
                sum(p.numel() for p in self.timeframe_embedding.parameters()) +
                sum(p.numel() for p in self.embedding_projection.parameters())
            )
        }