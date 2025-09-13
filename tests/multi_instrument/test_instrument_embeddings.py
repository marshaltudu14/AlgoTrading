"""
Tests for instrument embedding system
"""

import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import tempfile
import os

from src.feature_processing.instrument_embeddings import (
    InstrumentMetadata,
    InstrumentEmbedding,
    InstrumentEmbeddingLayer,
    InstrumentRegistry,
    InstrumentEmbeddingTrainer
)


class TestInstrumentMetadata:
    """Test InstrumentMetadata dataclass"""

    def test_instrument_metadata_creation(self):
        """Test creating instrument metadata"""
        metadata = InstrumentMetadata(
            symbol="AAPL",
            name="Apple Inc.",
            asset_class="equity",
            exchange="NASDAQ",
            currency="USD",
            tick_size=0.01,
            contract_size=1.0,
            trading_hours={"open": "09:30", "close": "16:00"}
        )

        assert metadata.symbol == "AAPL"
        assert metadata.name == "Apple Inc."
        assert metadata.asset_class == "equity"
        assert metadata.exchange == "NASDAQ"
        assert metadata.currency == "USD"
        assert metadata.tick_size == 0.01
        assert metadata.contract_size == 1.0
        assert metadata.trading_hours == {"open": "09:30", "close": "16:00"}
        assert isinstance(metadata.created_at, datetime)


class TestInstrumentEmbedding:
    """Test InstrumentEmbedding dataclass"""

    def test_instrument_embedding_creation(self):
        """Test creating instrument embedding"""
        metadata = InstrumentMetadata(
            symbol="AAPL",
            name="Apple Inc.",
            asset_class="equity",
            exchange="NASDAQ",
            currency="USD",
            tick_size=0.01,
            contract_size=1.0,
            trading_hours={"open": "09:30", "close": "16:00"}
        )

        embedding = np.random.randn(64)
        instrument_embedding = InstrumentEmbedding(
            symbol="AAPL",
            embedding=embedding,
            metadata=metadata,
            embedding_version="1.0",
            performance_metrics={"accuracy": 0.85, "sharpe_ratio": 1.2}
        )

        assert instrument_embedding.symbol == "AAPL"
        assert np.array_equal(instrument_embedding.embedding, embedding)
        assert instrument_embedding.metadata == metadata
        assert instrument_embedding.embedding_version == "1.0"
        assert instrument_embedding.performance_metrics == {"accuracy": 0.85, "sharpe_ratio": 1.2}


class TestInstrumentEmbeddingLayer:
    """Test InstrumentEmbeddingLayer neural network"""

    def test_embedding_layer_initialization(self):
        """Test initializing embedding layer"""
        layer = InstrumentEmbeddingLayer(num_instruments=10, embedding_dim=32)

        assert layer.num_instruments == 10
        assert layer.embedding_dim == 32
        assert layer.embedding.weight.shape == (10, 32)

    def test_forward_pass(self):
        """Test forward pass through embedding layer"""
        layer = InstrumentEmbeddingLayer(num_instruments=5, embedding_dim=16)
        instrument_ids = torch.tensor([0, 1, 2, 3, 4])

        result = layer(instrument_ids)

        assert 'embeddings' in result
        assert 'volatility' in result
        assert 'liquidity' in result
        assert 'trend' in result

        assert result['embeddings'].shape == (5, 16)
        assert result['volatility'].shape == (5, 1)
        assert result['liquidity'].shape == (5, 1)
        assert result['trend'].shape == (5, 1)

    def test_get_single_embedding(self):
        """Test getting embedding for single instrument"""
        layer = InstrumentEmbeddingLayer(num_instruments=3, embedding_dim=8)
        embedding = layer.get_embedding(1)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (8,)


class TestInstrumentRegistry:
    """Test InstrumentRegistry"""

    def test_registry_initialization(self):
        """Test initializing registry"""
        registry = InstrumentRegistry()

        assert registry.instruments == {}
        assert registry.embeddings == {}
        assert registry.symbol_to_id == {}
        assert registry.id_to_symbol == {}
        assert registry.next_id == 0

    def test_register_instrument(self):
        """Test registering an instrument"""
        registry = InstrumentRegistry()
        metadata = InstrumentMetadata(
            symbol="AAPL",
            name="Apple Inc.",
            asset_class="equity",
            exchange="NASDAQ",
            currency="USD",
            tick_size=0.01,
            contract_size=1.0,
            trading_hours={"open": "09:30", "close": "16:00"}
        )

        instrument_id = registry.register_instrument(metadata)

        assert instrument_id == 0
        assert "AAPL" in registry.instruments
        assert registry.symbol_to_id["AAPL"] == 0
        assert registry.id_to_symbol[0] == "AAPL"
        assert registry.next_id == 1

    def test_register_duplicate_instrument(self):
        """Test registering duplicate instrument"""
        registry = InstrumentRegistry()
        metadata = InstrumentMetadata(
            symbol="AAPL",
            name="Apple Inc.",
            asset_class="equity",
            exchange="NASDAQ",
            currency="USD",
            tick_size=0.01,
            contract_size=1.0,
            trading_hours={"open": "09:30", "close": "16:00"}
        )

        # Register first time
        instrument_id1 = registry.register_instrument(metadata)
        # Register second time
        instrument_id2 = registry.register_instrument(metadata)

        assert instrument_id1 == instrument_id2 == 0
        assert registry.next_id == 1  # Should not increment

    def test_update_embedding(self):
        """Test updating instrument embedding"""
        registry = InstrumentRegistry()
        metadata = InstrumentMetadata(
            symbol="AAPL",
            name="Apple Inc.",
            asset_class="equity",
            exchange="NASDAQ",
            currency="USD",
            tick_size=0.01,
            contract_size=1.0,
            trading_hours={"open": "09:30", "close": "16:00"}
        )

        registry.register_instrument(metadata)
        embedding = np.random.randn(32)

        registry.update_embedding("AAPL", embedding, {"accuracy": 0.85})

        assert "AAPL" in registry.embeddings
        assert np.array_equal(registry.embeddings["AAPL"].embedding, embedding)
        assert registry.embeddings["AAPL"].performance_metrics == {"accuracy": 0.85}

    def test_get_embedding_nonexistent(self):
        """Test getting embedding for non-existent instrument"""
        registry = InstrumentRegistry()
        embedding = registry.get_embedding("NONEXISTENT")

        assert embedding is None

    def test_similarity_computation(self):
        """Test computing similarity between instruments"""
        registry = InstrumentRegistry()
        metadata1 = InstrumentMetadata("AAPL", "Apple Inc.", "equity", "NASDAQ", "USD", 0.01, 1.0, {})
        metadata2 = InstrumentMetadata("MSFT", "Microsoft Corp.", "equity", "NASDAQ", "USD", 0.01, 1.0, {})

        registry.register_instrument(metadata1)
        registry.register_instrument(metadata2)

        # Create normalized embeddings for proper similarity calculation
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.9, 0.1, 0.0])
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        registry.update_embedding("AAPL", embedding1)
        registry.update_embedding("MSFT", embedding2)

        similarity = registry.compute_similarity("AAPL", "MSFT")

        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0

    def test_find_similar_instruments(self):
        """Test finding similar instruments"""
        registry = InstrumentRegistry()

        # Register multiple instruments
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        for symbol in symbols:
            metadata = InstrumentMetadata(symbol, f"{symbol} Inc.", "equity", "NASDAQ", "USD", 0.01, 1.0, {})
            registry.register_instrument(metadata)
            # Use controlled embeddings for predictable similarity scores
            np.random.seed(42 + hash(symbol) % 100)  # Deterministic but different per symbol
            embedding = np.random.randn(8)
            # Normalize embeddings to ensure proper similarity calculation
            embedding = embedding / np.linalg.norm(embedding)
            registry.update_embedding(symbol, embedding)

        similar = registry.find_similar_instruments("AAPL", top_k=3)

        assert len(similar) <= 3
        for symbol, similarity_score in similar:
            assert symbol != "AAPL"
            # Similarity should be between -1 and 1 for cosine similarity
            assert -1.0 <= similarity_score <= 1.0

    def test_list_instruments_by_asset_class(self):
        """Test listing instruments by asset class"""
        registry = InstrumentRegistry()

        # Register different asset classes
        equity_metadata = InstrumentMetadata("AAPL", "Apple Inc.", "equity", "NASDAQ", "USD", 0.01, 1.0, {})
        crypto_metadata = InstrumentMetadata("BTC", "Bitcoin", "crypto", "Binance", "USD", 0.01, 1.0, {})

        registry.register_instrument(equity_metadata)
        registry.register_instrument(crypto_metadata)

        equity_instruments = registry.list_instruments(asset_class="equity")
        crypto_instruments = registry.list_instruments(asset_class="crypto")
        all_instruments = registry.list_instruments()

        assert "AAPL" in equity_instruments
        assert "BTC" in crypto_instruments
        assert len(all_instruments) == 2

    def test_save_and_load_registry(self):
        """Test saving and loading registry"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "test_registry.json")
            registry = InstrumentRegistry(registry_path)

            # Register instruments and embeddings
            metadata = InstrumentMetadata("AAPL", "Apple Inc.", "equity", "NASDAQ", "USD", 0.01, 1.0, {})
            registry.register_instrument(metadata)
            embedding = np.random.randn(16)
            registry.update_embedding("AAPL", embedding)

            # Save registry
            registry.save_registry()

            # Create new registry and load
            new_registry = InstrumentRegistry(registry_path)
            new_registry.load_registry()

            # Verify loaded data
            assert "AAPL" in new_registry.instruments
            assert "AAPL" in new_registry.embeddings
            assert np.array_equal(new_registry.get_embedding("AAPL"), embedding)


class TestInstrumentEmbeddingTrainer:
    """Test InstrumentEmbeddingTrainer"""

    def test_trainer_initialization(self):
        """Test initializing trainer"""
        registry = InstrumentRegistry()
        trainer = InstrumentEmbeddingTrainer(registry, embedding_dim=32)

        assert trainer.registry == registry
        assert trainer.embedding_dim == 32
        assert trainer.embedding_layer is None

    def test_initialize_embeddings(self):
        """Test initializing embedding layer"""
        registry = InstrumentRegistry()
        trainer = InstrumentEmbeddingTrainer(registry, embedding_dim=16)

        trainer.initialize_embeddings(num_instruments=5)

        assert trainer.embedding_layer is not None
        assert trainer.embedding_layer.num_instruments == 5
        assert trainer.embedding_layer.embedding_dim == 16

    def test_train_embeddings(self):
        """Test training embeddings"""
        registry = InstrumentRegistry()
        trainer = InstrumentEmbeddingTrainer(registry, embedding_dim=8)

        # Register instruments
        symbols = ["AAPL", "MSFT", "GOOGL"]
        for symbol in symbols:
            metadata = InstrumentMetadata(symbol, f"{symbol} Inc.", "equity", "NASDAQ", "USD", 0.01, 1.0, {})
            registry.register_instrument(metadata)

        # Create mock market data
        market_data = {}
        for symbol in symbols:
            market_data[symbol] = pd.DataFrame({
                'open': np.random.randn(100),
                'high': np.random.randn(100),
                'low': np.random.randn(100),
                'close': np.random.randn(100),
                'volume': np.random.randn(100)
            })

        # Train embeddings
        results = trainer.train_embeddings(market_data, epochs=10, learning_rate=0.001)

        assert 'epochs_completed' in results
        assert 'final_loss' in results
        assert 'embedding_dim' in results
        assert 'num_instruments' in results
        assert results['epochs_completed'] == 10
        assert results['embedding_dim'] == 8
        assert results['num_instruments'] == 3

        # Verify embeddings were updated in registry
        for symbol in symbols:
            embedding = registry.get_embedding(symbol)
            assert embedding is not None
            assert embedding.shape == (8,)


class TestInstrumentEmbeddingIntegration:
    """Integration tests for instrument embedding system"""

    def test_multi_instrument_workflow(self):
        """Test complete multi-instrument workflow"""
        # Create registry
        registry = InstrumentRegistry()

        # Register diverse instruments
        instruments = [
            ("AAPL", "Apple Inc.", "equity", "NASDAQ", "USD"),
            ("MSFT", "Microsoft Corp.", "equity", "NASDAQ", "USD"),
            ("EURUSD", "EUR/USD", "forex", "Forex", "USD"),
            ("BTC", "Bitcoin", "crypto", "Binance", "USD"),
            ("GC", "Gold", "commodity", "COMEX", "USD"),
            ("CL", "Crude Oil", "commodity", "NYMEX", "USD")
        ]

        for symbol, name, asset_class, exchange, currency in instruments:
            metadata = InstrumentMetadata(
                symbol=symbol,
                name=name,
                asset_class=asset_class,
                exchange=exchange,
                currency=currency,
                tick_size=0.01,
                contract_size=1.0,
                trading_hours={"open": "09:30", "close": "16:00"}
            )
            registry.register_instrument(metadata)

        # Create trainer and train embeddings
        trainer = InstrumentEmbeddingTrainer(registry, embedding_dim=16)

        # Create mock market data
        market_data = {}
        for symbol, _, _, _, _ in instruments:
            market_data[symbol] = pd.DataFrame({
                'open': np.random.randn(50),
                'high': np.random.randn(50),
                'low': np.random.randn(50),
                'close': np.random.randn(50),
                'volume': np.random.randn(50)
            })

        results = trainer.train_embeddings(market_data, epochs=5)

        # Verify all instruments have embeddings
        for symbol, _, _, _, _ in instruments:
            embedding = registry.get_embedding(symbol)
            assert embedding is not None
            assert embedding.shape == (16,)

        # Test similarity between instruments
        similarity = registry.compute_similarity("AAPL", "MSFT")
        assert isinstance(similarity, float)

        # Test finding similar instruments
        similar = registry.find_similar_instruments("AAPL", top_k=2)
        assert len(similar) <= 2

        # Test listing by asset class
        equity_instruments = registry.list_instruments(asset_class="equity")
        assert len(equity_instruments) >= 1

        print("Multi-instrument workflow test completed successfully")

    def test_performance_with_many_instruments(self):
        """Test performance with many instruments (acceptance criteria #1)"""
        registry = InstrumentRegistry()

        # Register 10+ instruments (meets acceptance criteria)
        symbols = [f"INST_{i:02d}" for i in range(1, 12)]
        for symbol in symbols:
            metadata = InstrumentMetadata(
                symbol=symbol,
                name=f"Instrument {int(symbol.split('_')[1]):02d}",
                asset_class="equity",
                exchange="TEST",
                currency="USD",
                tick_size=0.01,
                contract_size=1.0,
                trading_hours={"open": "09:30", "close": "16:00"}
            )
            registry.register_instrument(metadata)

        # Verify all instruments registered
        assert len(registry.instruments) == 11

        # Train embeddings
        trainer = InstrumentEmbeddingTrainer(registry, embedding_dim=32)
        market_data = {symbol: pd.DataFrame(np.random.randn(50, 5)) for symbol in symbols}

        start_time = datetime.now()
        results = trainer.train_embeddings(market_data, epochs=3)
        end_time = datetime.now()

        training_time = (end_time - start_time).total_seconds()
        print(f"Training time for 11 instruments: {training_time:.2f} seconds")

        # Verify performance consistency
        assert results['num_instruments'] == 11
        assert training_time < 30.0  # Should complete within reasonable time

        # Test similarity computation performance
        start_time = datetime.now()
        for i in range(len(symbols) - 1):
            registry.compute_similarity(symbols[i], symbols[i + 1])
        end_time = datetime.now()

        similarity_time = (end_time - start_time).total_seconds()
        print(f"Similarity computation time for 10 pairs: {similarity_time:.2f} seconds")
        assert similarity_time < 1.0  # Should be fast


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    pytest.main([__file__, "-v"])