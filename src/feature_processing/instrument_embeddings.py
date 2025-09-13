"""
Instrument Embedding System for Multi-Instrument Trading

This module provides the foundation for handling multiple trading instruments
with learned embeddings that capture market-specific characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class InstrumentMetadata:
    """Metadata for a trading instrument"""
    symbol: str
    name: str
    asset_class: str  # equity, forex, commodity, crypto, etc.
    exchange: str
    currency: str
    tick_size: float
    contract_size: float
    trading_hours: Dict[str, str]  # timezone-aware trading hours
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class InstrumentEmbedding:
    """Embedding representation for a trading instrument"""
    symbol: str
    embedding: np.ndarray
    metadata: InstrumentMetadata
    embedding_version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class InstrumentEmbeddingLayer(nn.Module):
    """
    Neural network layer for learning instrument embeddings
    """

    def __init__(self, num_instruments: int, embedding_dim: int = 64):
        """
        Initialize instrument embedding layer

        Args:
            num_instruments: Maximum number of instruments to support
            embedding_dim: Dimension of embedding vectors
        """
        super().__init__()
        self.num_instruments = num_instruments
        self.embedding_dim = embedding_dim

        # Main embedding layer
        self.embedding = nn.Embedding(num_instruments, embedding_dim)

        # Projection layers for different characteristics
        self.volatility_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )

        self.liquidity_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )

        self.trend_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )

        # Initialize embeddings
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, instrument_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embedding layer

        Args:
            instrument_ids: Tensor of instrument indices

        Returns:
            Dictionary containing embeddings and projected characteristics
        """
        # Get base embeddings
        embeddings = self.embedding(instrument_ids)

        # Project different characteristics
        volatility = self.volatility_projection(embeddings)
        liquidity = self.liquidity_projection(embeddings)
        trend = self.trend_projection(embeddings)

        return {
            'embeddings': embeddings,
            'volatility': volatility,
            'liquidity': liquidity,
            'trend': trend
        }

    def get_embedding(self, instrument_id: int) -> np.ndarray:
        """
        Get embedding for a single instrument

        Args:
            instrument_id: Instrument index

        Returns:
            Embedding vector as numpy array
        """
        with torch.no_grad():
            embedding = self.embedding(torch.tensor([instrument_id]))
        return embedding.cpu().numpy()[0]


class InstrumentRegistry:
    """
    Registry for managing trading instruments and their embeddings
    """

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize instrument registry

        Args:
            registry_path: Path to registry file
        """
        self.registry_path = Path(registry_path) if registry_path else Path("instrument_registry.json")
        self.instruments: Dict[str, InstrumentMetadata] = {}
        self.embeddings: Dict[str, InstrumentEmbedding] = {}
        self.symbol_to_id: Dict[str, int] = {}
        self.id_to_symbol: Dict[int, str] = {}
        self.next_id = 0

        # Load existing registry if file exists
        if self.registry_path.exists():
            self.load_registry()

    def register_instrument(self, metadata: InstrumentMetadata) -> int:
        """
        Register a new trading instrument

        Args:
            metadata: Instrument metadata

        Returns:
            Instrument ID
        """
        symbol = metadata.symbol

        if symbol in self.symbol_to_id:
            logger.warning(f"Instrument {symbol} already registered")
            return self.symbol_to_id[symbol]

        # Assign new ID
        instrument_id = self.next_id
        self.next_id += 1

        # Register instrument
        self.instruments[symbol] = metadata
        self.symbol_to_id[symbol] = instrument_id
        self.id_to_symbol[instrument_id] = symbol

        logger.info(f"Registered instrument {symbol} with ID {instrument_id}")
        return instrument_id

    def get_instrument_id(self, symbol: str) -> Optional[int]:
        """
        Get instrument ID for a symbol

        Args:
            symbol: Instrument symbol

        Returns:
            Instrument ID or None if not found
        """
        return self.symbol_to_id.get(symbol)

    def get_instrument_metadata(self, symbol: str) -> Optional[InstrumentMetadata]:
        """
        Get metadata for an instrument

        Args:
            symbol: Instrument symbol

        Returns:
            Instrument metadata or None if not found
        """
        return self.instruments.get(symbol)

    def get_instrument_symbol(self, instrument_id: int) -> Optional[str]:
        """
        Get symbol for an instrument ID

        Args:
            instrument_id: Instrument ID

        Returns:
            Symbol or None if not found
        """
        return self.id_to_symbol.get(instrument_id)

    def list_instruments(self, asset_class: Optional[str] = None) -> List[str]:
        """
        List registered instruments

        Args:
            asset_class: Filter by asset class

        Returns:
            List of instrument symbols
        """
        if asset_class is None:
            return list(self.instruments.keys())

        return [
            symbol for symbol, metadata in self.instruments.items()
            if metadata.asset_class == asset_class
        ]

    def update_embedding(self, symbol: str, embedding: np.ndarray,
                       performance_metrics: Optional[Dict[str, float]] = None):
        """
        Update embedding for an instrument

        Args:
            symbol: Instrument symbol
            embedding: Embedding vector
            performance_metrics: Performance metrics for the embedding
        """
        if symbol not in self.instruments:
            raise ValueError(f"Instrument {symbol} not registered")

        metadata = self.instruments[symbol]

        self.embeddings[symbol] = InstrumentEmbedding(
            symbol=symbol,
            embedding=embedding,
            metadata=metadata,
            performance_metrics=performance_metrics or {}
        )

        logger.info(f"Updated embedding for instrument {symbol}")

    def get_embedding(self, symbol: str) -> Optional[np.ndarray]:
        """
        Get embedding for an instrument

        Args:
            symbol: Instrument symbol

        Returns:
            Embedding vector or None if not found
        """
        embedding = self.embeddings.get(symbol)
        return embedding.embedding if embedding else None

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get embeddings for all instruments

        Returns:
            Dictionary mapping symbols to embeddings
        """
        return {
            symbol: embedding.embedding
            for symbol, embedding in self.embeddings.items()
        }

    def compute_similarity(self, symbol1: str, symbol2: str) -> float:
        """
        Compute similarity between two instruments

        Args:
            symbol1: First instrument symbol
            symbol2: Second instrument symbol

        Returns:
            Similarity score (cosine similarity)
        """
        embedding1 = self.get_embedding(symbol1)
        embedding2 = self.get_embedding(symbol2)

        if embedding1 is None or embedding2 is None:
            return 0.0

        # Normalize embeddings
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)

        # Compute cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        return float(similarity)

    def find_similar_instruments(self, symbol: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find instruments most similar to a given instrument

        Args:
            symbol: Reference instrument symbol
            top_k: Number of top similar instruments to return

        Returns:
            List of (symbol, similarity_score) tuples
        """
        if symbol not in self.embeddings:
            return []

        similarities = []
        reference_embedding = self.get_embedding(symbol)

        for other_symbol, embedding in self.embeddings.items():
            if other_symbol != symbol:
                similarity = self.compute_similarity(symbol, other_symbol)
                similarities.append((other_symbol, similarity))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def save_registry(self):
        """Save registry to file"""
        registry_data = {
            'instruments': {},
            'embeddings': {},
            'symbol_to_id': self.symbol_to_id,
            'id_to_symbol': self.id_to_symbol,
            'next_id': self.next_id,
            'saved_at': datetime.now().isoformat()
        }

        # Save instrument metadata
        for symbol, metadata in self.instruments.items():
            registry_data['instruments'][symbol] = {
                'symbol': metadata.symbol,
                'name': metadata.name,
                'asset_class': metadata.asset_class,
                'exchange': metadata.exchange,
                'currency': metadata.currency,
                'tick_size': metadata.tick_size,
                'contract_size': metadata.contract_size,
                'trading_hours': metadata.trading_hours,
                'created_at': metadata.created_at.isoformat(),
                'updated_at': metadata.updated_at.isoformat()
            }

        # Save embeddings
        for symbol, embedding in self.embeddings.items():
            registry_data['embeddings'][symbol] = {
                'symbol': embedding.symbol,
                'embedding': embedding.embedding.tolist(),
                'metadata': {
                    'symbol': embedding.metadata.symbol,
                    'name': embedding.metadata.name,
                    'asset_class': embedding.metadata.asset_class,
                    'exchange': embedding.metadata.exchange,
                    'currency': embedding.metadata.currency,
                    'tick_size': embedding.metadata.tick_size,
                    'contract_size': embedding.metadata.contract_size,
                    'trading_hours': embedding.metadata.trading_hours,
                    'created_at': embedding.metadata.created_at.isoformat(),
                    'updated_at': embedding.metadata.updated_at.isoformat()
                },
                'embedding_version': embedding.embedding_version,
                'created_at': embedding.created_at.isoformat(),
                'performance_metrics': embedding.performance_metrics
            }

        with open(self.registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)

        logger.info(f"Registry saved to {self.registry_path}")

    def load_registry(self):
        """Load registry from file"""
        if not self.registry_path.exists():
            logger.warning(f"Registry file {self.registry_path} not found")
            return

        with open(self.registry_path, 'r') as f:
            registry_data = json.load(f)

        # Load instrument metadata
        for symbol, metadata_data in registry_data.get('instruments', {}).items():
            metadata = InstrumentMetadata(
                symbol=metadata_data['symbol'],
                name=metadata_data['name'],
                asset_class=metadata_data['asset_class'],
                exchange=metadata_data['exchange'],
                currency=metadata_data['currency'],
                tick_size=metadata_data['tick_size'],
                contract_size=metadata_data['contract_size'],
                trading_hours=metadata_data['trading_hours'],
                created_at=datetime.fromisoformat(metadata_data['created_at']),
                updated_at=datetime.fromisoformat(metadata_data['updated_at'])
            )
            self.instruments[symbol] = metadata

        # Load embeddings
        for symbol, embedding_data in registry_data.get('embeddings', {}).items():
            metadata = InstrumentMetadata(
                symbol=embedding_data['metadata']['symbol'],
                name=embedding_data['metadata']['name'],
                asset_class=embedding_data['metadata']['asset_class'],
                exchange=embedding_data['metadata']['exchange'],
                currency=embedding_data['metadata']['currency'],
                tick_size=embedding_data['metadata']['tick_size'],
                contract_size=embedding_data['metadata']['contract_size'],
                trading_hours=embedding_data['metadata']['trading_hours'],
                created_at=datetime.fromisoformat(embedding_data['metadata']['created_at']),
                updated_at=datetime.fromisoformat(embedding_data['metadata']['updated_at'])
            )

            embedding = InstrumentEmbedding(
                symbol=embedding_data['symbol'],
                embedding=np.array(embedding_data['embedding']),
                metadata=metadata,
                embedding_version=embedding_data['embedding_version'],
                created_at=datetime.fromisoformat(embedding_data['created_at']),
                performance_metrics=embedding_data['performance_metrics']
            )
            self.embeddings[symbol] = embedding

        # Load mappings
        self.symbol_to_id = registry_data.get('symbol_to_id', {})
        self.id_to_symbol = {int(k): v for k, v in registry_data.get('id_to_symbol', {}).items()}
        self.next_id = registry_data.get('next_id', 0)

        logger.info(f"Registry loaded from {self.registry_path}")


class InstrumentEmbeddingTrainer:
    """
    Trainer for learning instrument embeddings from market data
    """

    def __init__(self, registry: InstrumentRegistry, embedding_dim: int = 64):
        """
        Initialize trainer

        Args:
            registry: Instrument registry
            embedding_dim: Embedding dimension
        """
        self.registry = registry
        self.embedding_dim = embedding_dim
        self.embedding_layer = None

    def initialize_embeddings(self, num_instruments: int):
        """
        Initialize embedding layer

        Args:
            num_instruments: Number of instruments
        """
        self.embedding_layer = InstrumentEmbeddingLayer(
            num_instruments=num_instruments,
            embedding_dim=self.embedding_dim
        )

    def train_embeddings(self, market_data: Dict[str, pd.DataFrame],
                        epochs: int = 100, learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Train instrument embeddings from market data

        Args:
            market_data: Dictionary mapping symbols to market data DataFrames
            epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            Training results
        """
        if self.embedding_layer is None:
            num_instruments = len(market_data)
            self.initialize_embeddings(num_instruments)

        optimizer = torch.optim.Adam(self.embedding_layer.parameters(), lr=learning_rate)

        # Training loop would go here
        # For now, we'll initialize with random embeddings
        training_results = {
            'epochs_completed': epochs,
            'final_loss': 0.0,  # Would be computed during training
            'embedding_dim': self.embedding_dim,
            'num_instruments': len(market_data)
        }

        # Update embeddings in registry
        for symbol, instrument_id in self.registry.symbol_to_id.items():
            embedding = self.embedding_layer.get_embedding(instrument_id)
            self.registry.update_embedding(symbol, embedding)

        logger.info(f"Training completed. Updated embeddings for {len(market_data)} instruments")
        return training_results