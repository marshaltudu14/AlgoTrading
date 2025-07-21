"""
Unit tests for ExternalMemory module.

Tests the functionality of storing and retrieving memories,
similarity search, and memory management features.
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from src.memory.episodic_memory import ExternalMemory, MemoryEvent


class TestMemoryEvent:
    """Test cases for MemoryEvent dataclass."""
    
    def test_memory_event_creation(self):
        """Test MemoryEvent creation with different embedding types."""
        # Test with numpy array
        embedding_np = np.random.randn(10)
        outcome = {'reward': 1.5, 'action': 2}
        
        event = MemoryEvent(
            embedding=embedding_np,
            outcome=outcome,
            timestamp=1.0,
            importance=0.8,
            metadata={'type': 'test'}
        )
        
        assert isinstance(event.embedding, np.ndarray)
        assert event.outcome == outcome
        assert event.timestamp == 1.0
        assert event.importance == 0.8
        
        # Test with torch tensor
        embedding_torch = torch.randn(10)
        event_torch = MemoryEvent(
            embedding=embedding_torch,
            outcome=outcome,
            timestamp=2.0,
            importance=0.9,
            metadata={}
        )
        
        assert isinstance(event_torch.embedding, np.ndarray)
        assert event_torch.embedding.shape == (10,)


class TestExternalMemory:
    """Test cases for ExternalMemory class."""
    
    def test_external_memory_initialization(self):
        """Test ExternalMemory initialization with different parameters."""
        memory = ExternalMemory(
            max_memories=1000,
            embedding_dim=128,
            similarity_threshold=0.8,
            top_k_retrieval=3
        )
        
        assert memory.max_memories == 1000
        assert memory.embedding_dim == 128
        assert memory.similarity_threshold == 0.8
        assert memory.top_k_retrieval == 3
        assert len(memory.memories) == 0
        assert memory.embedding_matrix is None
    
    def test_store_memory_basic(self):
        """Test basic memory storage functionality."""
        memory = ExternalMemory(embedding_dim=10)
        
        # Store a memory
        embedding = np.random.randn(10)
        outcome = {'reward': 1.0, 'action': 1, 'profit': 100.0}
        
        memory.store(embedding, outcome, importance=0.8)
        
        assert len(memory.memories) == 1
        assert memory.total_stored == 1
        assert memory.memories[0].outcome == outcome
        assert abs(memory.memories[0].importance - 0.8) < 0.01  # Allow for small floating point differences
        assert memory.embedding_matrix is not None
        assert memory.embedding_matrix.shape == (1, 10)
    
    def test_store_memory_with_torch_tensor(self):
        """Test storing memory with PyTorch tensor embedding."""
        memory = ExternalMemory(embedding_dim=15)
        
        embedding = torch.randn(15)
        outcome = {'reward': -0.5, 'action': 0}
        
        memory.store(embedding, outcome)
        
        assert len(memory.memories) == 1
        assert isinstance(memory.memories[0].embedding, np.ndarray)
        assert memory.memories[0].embedding.shape == (15,)
    
    def test_store_memory_dimension_mismatch(self):
        """Test storing memory with mismatched embedding dimensions."""
        memory = ExternalMemory(embedding_dim=10)
        
        # Test with smaller dimension (should be padded)
        small_embedding = np.random.randn(5)
        memory.store(small_embedding, {'reward': 1.0})
        
        assert memory.memories[0].embedding.shape == (10,)
        
        # Test with larger dimension (should be truncated)
        large_embedding = np.random.randn(15)
        memory.store(large_embedding, {'reward': 2.0})
        
        assert memory.memories[1].embedding.shape == (10,)
    
    def test_retrieve_memory_empty(self):
        """Test retrieving from empty memory."""
        memory = ExternalMemory(embedding_dim=10)
        
        query = np.random.randn(10)
        results = memory.retrieve(query)
        
        assert len(results) == 0
    
    def test_retrieve_memory_basic(self):
        """Test basic memory retrieval functionality."""
        memory = ExternalMemory(embedding_dim=10, similarity_threshold=0.0, top_k_retrieval=3)
        
        # Store some memories
        embeddings = [np.random.randn(10) for _ in range(5)]
        outcomes = [{'reward': i, 'action': i % 3} for i in range(5)]
        
        for i, (emb, out) in enumerate(zip(embeddings, outcomes)):
            memory.store(emb, out, importance=0.5 + i * 0.1)
        
        # Retrieve using one of the stored embeddings
        query = embeddings[2]  # Should be most similar to itself
        results = memory.retrieve(query)
        
        assert len(results) <= 3  # top_k_retrieval
        assert len(results) > 0
        
        # Check that results are sorted by similarity (descending)
        similarities = [sim for _, sim in results]
        assert similarities == sorted(similarities, reverse=True)
        
        # The most similar should be the exact match
        best_match, best_similarity = results[0]
        assert best_similarity > 0.99  # Should be very close to 1.0
    
    def test_retrieve_memory_with_similarity_threshold(self):
        """Test memory retrieval with similarity threshold."""
        memory = ExternalMemory(embedding_dim=10, similarity_threshold=0.8, top_k_retrieval=5)
        
        # Store memories with very different embeddings
        memory.store(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), {'reward': 1.0})
        memory.store(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), {'reward': 2.0})
        memory.store(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), {'reward': 3.0})
        
        # Query with first embedding (should only match itself)
        query = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        results = memory.retrieve(query)
        
        # Should only return memories above similarity threshold
        assert len(results) == 1  # Only exact match should be above 0.8
        assert results[0][1] > 0.99
    
    def test_memory_capacity_and_eviction(self):
        """Test memory capacity limits and eviction policy."""
        memory = ExternalMemory(max_memories=3, embedding_dim=5)
        
        # Store more memories than capacity
        for i in range(5):
            embedding = np.random.randn(5)
            outcome = {'reward': i}
            importance = 0.1 + i * 0.2  # Increasing importance
            memory.store(embedding, outcome, importance=importance)
        
        # Should only keep max_memories
        assert len(memory.memories) == 3
        
        # Should keep the most important ones (accounting for decay)
        importances = [mem.importance for mem in memory.memories]
        # The minimum importance should be higher than the lowest initial importance (0.1)
        # but may be affected by decay, so we check it's at least above the very lowest
        assert min(importances) > 0.4  # Should have evicted low importance memories
    
    def test_importance_decay(self):
        """Test importance decay functionality."""
        memory = ExternalMemory(embedding_dim=5, importance_decay=0.9)
        
        # Store a memory with high importance
        embedding = np.random.randn(5)
        memory.store(embedding, {'reward': 1.0}, importance=1.0)
        
        initial_importance = memory.memories[0].importance
        
        # Store another memory (should trigger decay)
        memory.store(np.random.randn(5), {'reward': 2.0}, importance=0.5)
        
        # First memory's importance should have decayed
        assert memory.memories[0].importance < initial_importance
        assert memory.memories[0].importance == initial_importance * 0.9
    
    def test_memory_statistics(self):
        """Test memory statistics functionality."""
        memory = ExternalMemory(max_memories=10, embedding_dim=5)
        
        # Empty memory stats
        stats = memory.get_memory_statistics()
        assert stats['total_memories'] == 0
        assert stats['memory_utilization'] == 0.0
        
        # Store some memories
        for i in range(3):
            embedding = np.random.randn(5)
            memory.store(embedding, {'reward': i}, importance=0.5 + i * 0.2)
        
        # Retrieve some memories
        query = np.random.randn(5)
        memory.retrieve(query)
        
        stats = memory.get_memory_statistics()
        assert stats['total_memories'] == 3
        assert stats['total_stored'] == 3
        assert stats['total_retrieved'] >= 0  # May be 0 if no memories meet similarity threshold
        assert stats['memory_utilization'] == 0.3  # 3/10
        assert 'avg_importance' in stats
        assert 'max_importance' in stats
        assert 'min_importance' in stats
    
    def test_clear_memory(self):
        """Test memory clearing functionality."""
        memory = ExternalMemory(embedding_dim=5)
        
        # Store some memories
        for i in range(3):
            memory.store(np.random.randn(5), {'reward': i})
        
        assert len(memory.memories) == 3
        
        # Clear memory
        memory.clear_memory()
        
        assert len(memory.memories) == 0
        assert memory.embedding_matrix is None
    
    def test_save_and_load_memories(self):
        """Test memory persistence functionality."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Create memory and store some data
            memory1 = ExternalMemory(embedding_dim=5, max_memories=10)
            
            embeddings = [np.random.randn(5) for _ in range(3)]
            outcomes = [{'reward': i, 'action': i} for i in range(3)]
            
            for emb, out in zip(embeddings, outcomes):
                memory1.store(emb, out, importance=0.8)
            
            # Save memories
            memory1.save_memories(tmp_path)
            
            # Create new memory instance and load
            memory2 = ExternalMemory(embedding_dim=5, max_memories=10)
            memory2.load_memories(tmp_path)
            
            # Check that memories were loaded correctly
            assert len(memory2.memories) == 3
            assert memory2.total_stored == 3
            
            # Check that embeddings and outcomes match
            for i in range(3):
                assert np.allclose(memory2.memories[i].embedding, memory1.memories[i].embedding)
                assert memory2.memories[i].outcome == memory1.memories[i].outcome
                
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_retrieve_with_custom_parameters(self):
        """Test retrieval with custom top_k and similarity parameters."""
        memory = ExternalMemory(embedding_dim=5, similarity_threshold=0.5, top_k_retrieval=2)
        
        # Store several memories
        for i in range(5):
            embedding = np.random.randn(5)
            memory.store(embedding, {'reward': i})
        
        query = np.random.randn(5)
        
        # Test custom top_k
        results_k3 = memory.retrieve(query, top_k=3)
        results_k1 = memory.retrieve(query, top_k=1)
        
        assert len(results_k3) <= 3
        assert len(results_k1) <= 1
        
        # Test custom similarity threshold
        results_high_sim = memory.retrieve(query, min_similarity=0.9)
        results_low_sim = memory.retrieve(query, min_similarity=0.1)
        
        assert len(results_high_sim) <= len(results_low_sim)


if __name__ == "__main__":
    pytest.main([__file__])
