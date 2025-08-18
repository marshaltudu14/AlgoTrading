#!/usr/bin/env python3
"""
Advanced Data Feeding Strategy for RL Training.
Implements optimal data feeding strategies to help RL models learn market dynamics without overfitting.
"""

import numpy as np
import pandas as pd
import random
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class FeedingStrategy(Enum):
    """Different data feeding strategies."""
    SEQUENTIAL = "sequential"           # Feed data in chronological order
    RANDOM = "random"                  # Random sampling from entire dataset
    CURRICULUM = "curriculum"          # Start simple, gradually increase complexity
    BALANCED = "balanced"              # Balance between different market conditions
    ADAPTIVE = "adaptive"              # Adapt strategy based on learning progress
    TEMPORAL_BLOCKS = "temporal_blocks" # Use temporal blocks to maintain some sequence
    MARKET_REGIME_AWARE = "market_regime_aware"  # Focus on different market regimes

@dataclass
class DataSegment:
    """Represents a segment of market data."""
    start_idx: int
    end_idx: int
    complexity_score: float
    volatility: float
    trend_strength: float
    market_regime: str
    performance_weight: float = 1.0

class DataFeedingStrategyManager:
    """
    Manages optimal data feeding strategies for RL training.
    
    The goal is to help the RL model learn market dynamics effectively while avoiding overfitting.
    Different strategies are used based on training progress and model performance.
    """
    
    def __init__(self, data: pd.DataFrame, lookback_window: int = 20, episode_length: int = 500):
        self.data = data
        self.lookback_window = lookback_window
        self.episode_length = episode_length
        self.total_length = len(data)
        
        # Analyze data characteristics
        self.data_segments = self._analyze_data_segments()
        self.market_regimes = self._identify_market_regimes()
        
        # Training progress tracking
        self.training_episodes = 0
        self.performance_history = []
        self.current_strategy = FeedingStrategy.CURRICULUM
        self.strategy_performance = {strategy: [] for strategy in FeedingStrategy}
        
        # Strategy parameters
        self.curriculum_progress = 0.0  # 0.0 to 1.0
        self.complexity_threshold = 0.3  # Start with low complexity
        self.adaptation_frequency = 100  # Evaluate strategy every N episodes
        
        logger.info(f"📊 Data Feeding Strategy Manager initialized")
        logger.info(f"   Total data length: {self.total_length}")
        logger.info(f"   Data segments: {len(self.data_segments)}")
        logger.info(f"   Market regimes identified: {len(set(seg.market_regime for seg in self.data_segments))}")
        
    def _analyze_data_segments(self) -> List[DataSegment]:
        """Analyze data and create segments with different characteristics."""
        segments = []

        # Handle small datasets
        if self.total_length < 100:
            logger.info(f"Small dataset ({self.total_length} rows), creating single segment")
            segments.append(DataSegment(
                start_idx=0,
                end_idx=self.total_length,
                complexity_score=0.5,
                volatility=0.1,
                trend_strength=0.1,
                market_regime="unknown"
            ))
            return segments

        segment_size = max(100, self.episode_length)  # Minimum segment size

        for i in range(0, self.total_length - segment_size, segment_size // 2):  # 50% overlap
            end_idx = min(i + segment_size, self.total_length)
            segment_data = self.data.iloc[i:end_idx]
            
            # Calculate segment characteristics
            complexity_score = self._calculate_complexity(segment_data)
            volatility = self._calculate_volatility(segment_data)
            trend_strength = self._calculate_trend_strength(segment_data)
            market_regime = self._classify_market_regime(segment_data)
            
            segments.append(DataSegment(
                start_idx=i,
                end_idx=end_idx,
                complexity_score=complexity_score,
                volatility=volatility,
                trend_strength=trend_strength,
                market_regime=market_regime
            ))
        
        return segments
    
    def _calculate_complexity(self, data: pd.DataFrame) -> float:
        """Calculate complexity score for a data segment (0-1)."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return 0.5
            
            # Use coefficient of variation as complexity measure
            cv_scores = []
            for col in numeric_data.columns:
                if numeric_data[col].std() > 0:
                    cv = numeric_data[col].std() / (abs(numeric_data[col].mean()) + 1e-8)
                    cv_scores.append(cv)
            
            if cv_scores:
                complexity = np.mean(cv_scores)
                return min(1.0, complexity / 2.0)  # Normalize to 0-1
            return 0.5
        except:
            return 0.5
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility for a data segment."""
        try:
            # Assume first numeric column is price-related
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
                returns = data[price_col].pct_change().dropna()
                return returns.std() if len(returns) > 1 else 0.0
            return 0.0
        except:
            return 0.0
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength for a data segment."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
                x = np.arange(len(data))
                slope = np.polyfit(x, data[price_col], 1)[0]
                return abs(slope)
            return 0.0
        except:
            return 0.0
    
    def _classify_market_regime(self, data: pd.DataFrame) -> str:
        """Classify market regime for a data segment."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return "unknown"
            
            price_col = numeric_cols[0]
            returns = data[price_col].pct_change().dropna()
            
            if len(returns) < 5:
                return "unknown"
            
            volatility = returns.std()
            trend = np.polyfit(range(len(returns)), returns.cumsum(), 1)[0]
            
            if volatility > 0.02:  # High volatility threshold
                return "high_volatility"
            elif trend > 0.001:
                return "uptrend"
            elif trend < -0.001:
                return "downtrend"
            else:
                return "sideways"
        except:
            return "unknown"
    
    def _identify_market_regimes(self) -> Dict[str, List[DataSegment]]:
        """Group segments by market regime."""
        regimes = {}
        for segment in self.data_segments:
            regime = segment.market_regime
            if regime not in regimes:
                regimes[regime] = []
            regimes[regime].append(segment)
        return regimes
    
    def get_next_episode_data(self, performance_metrics: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
        """
        Get the start and end indices for the next training episode.
        
        Args:
            performance_metrics: Recent performance metrics to adapt strategy
            
        Returns:
            Tuple of (start_idx, end_idx) for the episode
        """
        self.training_episodes += 1
        
        # Update performance tracking
        if performance_metrics:
            self.performance_history.append(performance_metrics)
            self.strategy_performance[self.current_strategy].append(performance_metrics.get('total_reward', 0.0))
        
        # Adapt strategy if needed
        if self.training_episodes % self.adaptation_frequency == 0:
            self._adapt_strategy()
        
        # Get episode data based on current strategy
        if self.current_strategy == FeedingStrategy.SEQUENTIAL:
            return self._get_sequential_episode()
        elif self.current_strategy == FeedingStrategy.RANDOM:
            return self._get_random_episode()
        elif self.current_strategy == FeedingStrategy.CURRICULUM:
            return self._get_curriculum_episode()
        elif self.current_strategy == FeedingStrategy.BALANCED:
            return self._get_balanced_episode()
        elif self.current_strategy == FeedingStrategy.ADAPTIVE:
            return self._get_adaptive_episode()
        elif self.current_strategy == FeedingStrategy.TEMPORAL_BLOCKS:
            return self._get_temporal_blocks_episode()
        elif self.current_strategy == FeedingStrategy.MARKET_REGIME_AWARE:
            return self._get_market_regime_aware_episode()
        else:
            return self._get_random_episode()  # Fallback
    
    def _get_sequential_episode(self) -> Tuple[int, int]:
        """Get episode data in sequential order."""
        max_start = self.total_length - self.episode_length - self.lookback_window
        if max_start <= 0:
            return 0, min(self.episode_length, self.total_length)
        
        # Cycle through data sequentially
        start_idx = (self.training_episodes * (self.episode_length // 4)) % max_start
        end_idx = min(start_idx + self.episode_length, self.total_length)
        
        return start_idx, end_idx
    
    def _get_random_episode(self) -> Tuple[int, int]:
        """Get random episode data."""
        max_start = self.total_length - self.episode_length - self.lookback_window
        if max_start <= 0:
            return 0, min(self.episode_length, self.total_length)
        
        start_idx = random.randint(0, max_start)
        end_idx = min(start_idx + self.episode_length, self.total_length)
        
        return start_idx, end_idx
    
    def _get_curriculum_episode(self) -> Tuple[int, int]:
        """Get episode data using curriculum learning (start simple, increase complexity)."""
        # Update curriculum progress
        self.curriculum_progress = min(1.0, self.training_episodes / 1000.0)  # Full curriculum over 1000 episodes
        self.complexity_threshold = 0.2 + 0.6 * self.curriculum_progress  # 0.2 to 0.8
        
        # Filter segments by complexity
        suitable_segments = [seg for seg in self.data_segments 
                           if seg.complexity_score <= self.complexity_threshold]
        
        if not suitable_segments:
            suitable_segments = self.data_segments  # Fallback to all segments

        # Handle empty data segments case
        if not suitable_segments:
            logger.info("No data segments available for curriculum episode, using random episode")
            return self._get_random_episode()

        # Select random segment from suitable ones
        segment = random.choice(suitable_segments)
        start_idx = random.randint(segment.start_idx, 
                                 max(segment.start_idx, segment.end_idx - self.episode_length))
        end_idx = min(start_idx + self.episode_length, segment.end_idx)
        
        logger.debug(f"Curriculum episode: complexity_threshold={self.complexity_threshold:.3f}, "
                    f"selected_complexity={segment.complexity_score:.3f}")
        
        return start_idx, end_idx
    
    def _get_balanced_episode(self) -> Tuple[int, int]:
        """Get episode data balanced across different market conditions."""
        # Ensure we sample from different market regimes
        regime_counts = {regime: 0 for regime in self.market_regimes.keys()}
        
        # Count recent regime usage (last 20 episodes)
        recent_episodes = min(20, len(self.performance_history))
        if recent_episodes > 0:
            # This is simplified - in practice, you'd track which regime each episode used
            pass
        
        # Select underrepresented regime
        min_count_regime = min(regime_counts.keys(), key=lambda x: len(self.market_regimes[x]))
        suitable_segments = self.market_regimes.get(min_count_regime, self.data_segments)
        
        if not suitable_segments:
            suitable_segments = self.data_segments

        # Handle empty data segments case
        if not suitable_segments:
            logger.info("No data segments available for balanced episode, using random episode")
            return self._get_random_episode()

        segment = random.choice(suitable_segments)
        start_idx = random.randint(segment.start_idx, 
                                 max(segment.start_idx, segment.end_idx - self.episode_length))
        end_idx = min(start_idx + self.episode_length, segment.end_idx)
        
        return start_idx, end_idx
    
    def _get_adaptive_episode(self) -> Tuple[int, int]:
        """Get episode data using adaptive strategy based on recent performance."""
        if len(self.performance_history) < 10:
            return self._get_curriculum_episode()  # Start with curriculum
        
        # Analyze recent performance
        recent_performance = np.mean([p.get('total_reward', 0.0) for p in self.performance_history[-10:]])
        
        if recent_performance < 0:  # Poor performance
            # Focus on simpler segments
            suitable_segments = [seg for seg in self.data_segments if seg.complexity_score < 0.5]
        else:  # Good performance
            # Challenge with more complex segments
            suitable_segments = [seg for seg in self.data_segments if seg.complexity_score > 0.5]
        
        if not suitable_segments:
            suitable_segments = self.data_segments
        
        segment = random.choice(suitable_segments)
        start_idx = random.randint(segment.start_idx, 
                                 max(segment.start_idx, segment.end_idx - self.episode_length))
        end_idx = min(start_idx + self.episode_length, segment.end_idx)
        
        return start_idx, end_idx
    
    def _get_temporal_blocks_episode(self) -> Tuple[int, int]:
        """Get episode data using temporal blocks to maintain some sequence."""
        block_size = self.episode_length * 3  # Larger blocks
        num_blocks = max(1, self.total_length // block_size)
        
        # Select a random block
        block_idx = random.randint(0, num_blocks - 1)
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, self.total_length)
        
        # Random position within the block
        max_start = block_end - self.episode_length - self.lookback_window
        if max_start <= block_start:
            start_idx = block_start
        else:
            start_idx = random.randint(block_start, max_start)
        
        end_idx = min(start_idx + self.episode_length, block_end)
        
        return start_idx, end_idx
    
    def _get_market_regime_aware_episode(self) -> Tuple[int, int]:
        """Get episode data focusing on specific market regimes."""
        # Cycle through different market regimes
        regime_names = list(self.market_regimes.keys())
        if not regime_names:
            return self._get_random_episode()
        
        current_regime = regime_names[self.training_episodes % len(regime_names)]
        suitable_segments = self.market_regimes[current_regime]
        
        if not suitable_segments:
            logger.warning("No suitable segments for adaptive episode, using random episode")
            return self._get_random_episode()

        segment = random.choice(suitable_segments)
        start_idx = random.randint(segment.start_idx, 
                                 max(segment.start_idx, segment.end_idx - self.episode_length))
        end_idx = min(start_idx + self.episode_length, segment.end_idx)
        
        logger.debug(f"Market regime aware episode: regime={current_regime}")
        
        return start_idx, end_idx
    
    def _adapt_strategy(self):
        """Adapt the feeding strategy based on performance."""
        if len(self.performance_history) < self.adaptation_frequency:
            return
        
        # Evaluate performance of current strategy
        current_performance = np.mean(self.strategy_performance[self.current_strategy][-10:]) if self.strategy_performance[self.current_strategy] else 0.0
        
        # Try different strategies if performance is poor
        if current_performance < 0:
            # Switch to curriculum learning for poor performance
            if self.current_strategy != FeedingStrategy.CURRICULUM:
                self.current_strategy = FeedingStrategy.CURRICULUM
                logger.info(f"🔄 Switched to CURRICULUM feeding strategy due to poor performance")
        elif current_performance > 0.5:
            # Switch to more challenging strategies for good performance
            if self.current_strategy == FeedingStrategy.CURRICULUM:
                self.current_strategy = FeedingStrategy.ADAPTIVE
                logger.info(f"🔄 Switched to ADAPTIVE feeding strategy due to good performance")
        
        # Log strategy performance
        logger.info(f"📊 Strategy adaptation: {self.current_strategy.value}")
        logger.info(f"   Current performance: {current_performance:.4f}")
        logger.info(f"   Training episodes: {self.training_episodes}")
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of data feeding strategy performance."""
        return {
            'current_strategy': self.current_strategy.value,
            'training_episodes': self.training_episodes,
            'curriculum_progress': self.curriculum_progress,
            'complexity_threshold': self.complexity_threshold,
            'total_segments': len(self.data_segments),
            'market_regimes': list(self.market_regimes.keys()),
            'strategy_performance': {
                strategy.value: np.mean(performances) if performances else 0.0
                for strategy, performances in self.strategy_performance.items()
            }
        }
