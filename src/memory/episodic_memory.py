"""
External Memory Module for Autonomous Trading Agents

This module provides episodic memory capabilities, allowing agents to store
and retrieve significant past events for learning and decision making.
The memory system uses vector similarity search to find relevant past experiences.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass
import pickle
import os
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryEvent:
    """
    Represents a single memory event stored in external memory.
    """
    embedding: np.ndarray  # Vector representation of the event
    outcome: Dict[str, Any]  # Outcome information (reward, action, etc.)
    timestamp: float  # When the event occurred
    importance: float  # Importance score for memory prioritization
    metadata: Dict[str, Any]  # Additional metadata about the event
    
    def __post_init__(self):
        """Ensure embedding is a numpy array."""
        if isinstance(self.embedding, torch.Tensor):
            self.embedding = self.embedding.detach().cpu().numpy()
        elif not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)


class ExternalMemory:
    """
    External Memory system for autonomous trading agents.
    
    This class provides a "notebook" where the agent can store and retrieve
    significant past events. It uses vector similarity search to find
    relevant memories based on the current state.
    
    Key features:
    - Store events with vector embeddings and outcomes
    - Retrieve similar past events using cosine similarity
    - Memory prioritization based on importance scores
    - Efficient similarity search using numpy operations
    - Memory persistence for long-term learning
    """
    
    def __init__(
        self,
        max_memories: int = 10000,
        embedding_dim: int = 512,
        similarity_threshold: float = 0.7,
        top_k_retrieval: int = 5,
        importance_decay: float = 0.99,
        memory_file: Optional[str] = None
    ):
        """
        Initialize the External Memory system.
        
        Args:
            max_memories: Maximum number of memories to store
            embedding_dim: Dimension of event embeddings
            similarity_threshold: Minimum similarity for memory retrieval
            top_k_retrieval: Number of top similar memories to retrieve
            importance_decay: Decay factor for memory importance over time
            memory_file: File path for persistent memory storage
        """
        self.max_memories = max_memories
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.top_k_retrieval = top_k_retrieval
        self.importance_decay = importance_decay
        self.memory_file = memory_file
        
        # Memory storage
        self.memories: List[MemoryEvent] = []
        self.embedding_matrix: Optional[np.ndarray] = None
        
        # Statistics
        self.total_stored = 0
        self.total_retrieved = 0
        
        # Load existing memories if file provided
        if memory_file and os.path.exists(memory_file):
            self.load_memories(memory_file)
    
    def store(
        self,
        event_embedding: Union[np.ndarray, torch.Tensor],
        outcome: Dict[str, Any],
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store a new memory event.
        
        Args:
            event_embedding: Vector representation of the event
            outcome: Outcome information (reward, action, profit, etc.)
            importance: Importance score for this memory (0.0 to 1.0)
            metadata: Additional metadata about the event
        """
        # Convert to numpy array if needed
        if isinstance(event_embedding, torch.Tensor):
            event_embedding = event_embedding.detach().cpu().numpy()
        elif not isinstance(event_embedding, np.ndarray):
            event_embedding = np.array(event_embedding)
        
        # Ensure correct dimensionality
        if event_embedding.shape[-1] != self.embedding_dim:
            logger.warning(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {event_embedding.shape[-1]}")
            # Pad or truncate as needed
            if event_embedding.shape[-1] < self.embedding_dim:
                padding = np.zeros(self.embedding_dim - event_embedding.shape[-1])
                event_embedding = np.concatenate([event_embedding, padding])
            else:
                event_embedding = event_embedding[:self.embedding_dim]
        
        # Normalize embedding for cosine similarity
        norm = np.linalg.norm(event_embedding)
        if norm > 0:
            event_embedding = event_embedding / norm
        
        # Create memory event
        memory_event = MemoryEvent(
            embedding=event_embedding,
            outcome=outcome,
            timestamp=self.total_stored,  # Use counter as timestamp
            importance=importance,
            metadata=metadata or {}
        )
        
        # Add to memory
        self.memories.append(memory_event)
        self.total_stored += 1
        
        # Apply memory limit with importance-based eviction
        if len(self.memories) > self.max_memories:
            self._evict_least_important_memory()
        
        # Update embedding matrix for efficient similarity search
        self._update_embedding_matrix()
        
        # Apply importance decay to existing memories
        self._apply_importance_decay()
        
        logger.debug(f"Stored memory event with importance {importance:.3f}. Total memories: {len(self.memories)}")
    
    def retrieve(
        self,
        current_state_embedding: Union[np.ndarray, torch.Tensor],
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None
    ) -> List[Tuple[MemoryEvent, float]]:
        """
        Retrieve the most similar past events from memory.
        
        Args:
            current_state_embedding: Vector representation of current state
            top_k: Number of top memories to retrieve (defaults to self.top_k_retrieval)
            min_similarity: Minimum similarity threshold (defaults to self.similarity_threshold)
            
        Returns:
            List of tuples (memory_event, similarity_score) sorted by similarity
        """
        if not self.memories:
            return []
        
        # Convert to numpy array if needed
        if isinstance(current_state_embedding, torch.Tensor):
            current_state_embedding = current_state_embedding.detach().cpu().numpy()
        elif not isinstance(current_state_embedding, np.ndarray):
            current_state_embedding = np.array(current_state_embedding)
        
        # Flatten if needed (remove batch dimension)
        if len(current_state_embedding.shape) > 1:
            current_state_embedding = current_state_embedding.flatten()

        # Ensure correct dimensionality
        if current_state_embedding.shape[-1] != self.embedding_dim:
            if current_state_embedding.shape[-1] < self.embedding_dim:
                padding = np.zeros(self.embedding_dim - current_state_embedding.shape[-1])
                current_state_embedding = np.concatenate([current_state_embedding, padding])
            else:
                current_state_embedding = current_state_embedding[:self.embedding_dim]
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(current_state_embedding)
        if norm > 0:
            current_state_embedding = current_state_embedding / norm
        
        # Use parameters or defaults
        top_k = top_k or self.top_k_retrieval
        min_similarity = min_similarity or self.similarity_threshold
        
        # Compute similarities using vectorized operations
        similarities = np.dot(self.embedding_matrix, current_state_embedding)
        
        # Apply importance weighting to similarities
        importance_weights = np.array([memory.importance for memory in self.memories])
        weighted_similarities = similarities * importance_weights
        
        # Get indices of top-k most similar memories
        top_indices = np.argsort(weighted_similarities)[::-1][:top_k]
        
        # Filter by minimum similarity threshold
        results = []
        for idx in top_indices:
            similarity = similarities[idx]  # Use original similarity, not weighted
            if similarity >= min_similarity:
                results.append((self.memories[idx], float(similarity)))
        
        self.total_retrieved += len(results)
        
        logger.debug(f"Retrieved {len(results)} memories with similarity >= {min_similarity:.3f}")
        
        return results
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        if not self.memories:
            return {
                'total_memories': 0,
                'total_stored': self.total_stored,
                'total_retrieved': self.total_retrieved,
                'avg_importance': 0.0,
                'memory_utilization': 0.0
            }
        
        importances = [memory.importance for memory in self.memories]
        
        return {
            'total_memories': len(self.memories),
            'total_stored': self.total_stored,
            'total_retrieved': self.total_retrieved,
            'avg_importance': np.mean(importances),
            'max_importance': np.max(importances),
            'min_importance': np.min(importances),
            'memory_utilization': len(self.memories) / self.max_memories,
            'embedding_dim': self.embedding_dim
        }
    
    def clear_memory(self) -> None:
        """Clear all stored memories."""
        self.memories.clear()
        self.embedding_matrix = None
        logger.info("Cleared all memories")
    
    def save_memories(self, filepath: str) -> None:
        """Save memories to file for persistence."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'memories': self.memories,
                    'total_stored': self.total_stored,
                    'total_retrieved': self.total_retrieved,
                    'config': {
                        'max_memories': self.max_memories,
                        'embedding_dim': self.embedding_dim,
                        'similarity_threshold': self.similarity_threshold,
                        'top_k_retrieval': self.top_k_retrieval,
                        'importance_decay': self.importance_decay
                    }
                }, f)
            logger.info(f"Saved {len(self.memories)} memories to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
    
    def load_memories(self, filepath: str) -> None:
        """Load memories from file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.memories = data['memories']
            self.total_stored = data.get('total_stored', len(self.memories))
            self.total_retrieved = data.get('total_retrieved', 0)
            
            # Update embedding matrix
            self._update_embedding_matrix()
            
            logger.info(f"Loaded {len(self.memories)} memories from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
    
    def _update_embedding_matrix(self) -> None:
        """Update the embedding matrix for efficient similarity search."""
        if self.memories:
            self.embedding_matrix = np.vstack([memory.embedding for memory in self.memories])
        else:
            self.embedding_matrix = None
    
    def _evict_least_important_memory(self) -> None:
        """Remove the least important memory when at capacity."""
        if not self.memories:
            return
        
        # Find memory with lowest importance
        min_importance_idx = min(range(len(self.memories)), 
                               key=lambda i: self.memories[i].importance)
        
        evicted_memory = self.memories.pop(min_importance_idx)
        logger.debug(f"Evicted memory with importance {evicted_memory.importance:.3f}")
    
    def _apply_importance_decay(self) -> None:
        """Apply decay to importance scores of existing memories."""
        for memory in self.memories:
            memory.importance *= self.importance_decay
