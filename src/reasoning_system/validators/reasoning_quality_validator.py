#!/usr/bin/env python3
"""
Reasoning Quality Validator
==========================

Validates the quality of generated reasoning text to ensure it meets production standards.
Detects robotic phrases, validates risk-reward ratios, checks consistency, and measures diversity.

Key Features:
- Detects and flags robotic phrases that reduce naturalness
- Validates risk-reward ratios are mathematically sound (0.1:1 to 5:1)
- Checks consistency between decision and supporting analysis
- Measures content diversity to prevent excessive repetition
- Provides actionable feedback for quality improvements
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReasoningQuality:
    """Data class for reasoning quality assessment."""
    naturalness_score: float  # 0.0 to 1.0
    diversity_score: float    # 0.0 to 1.0
    accuracy_score: float     # 0.0 to 1.0
    consistency_score: float  # 0.0 to 1.0
    issues: List[str]
    overall_score: float


class ReasoningQualityValidator:
    """
    Validates reasoning quality across multiple dimensions.
    """
    
    def __init__(self):
        """Initialize the quality validator."""
        self.robotic_phrases = self._load_robotic_phrases()
        self.consistency_patterns = self._load_consistency_patterns()
        self.phrase_usage_tracker = {}
        
        logger.info("ReasoningQualityValidator initialized")
    
    def validate_reasoning(self, reasoning: Dict[str, str]) -> ReasoningQuality:
        """
        Validate reasoning quality across all dimensions.
        
        Args:
            reasoning: Dictionary with reasoning columns
            
        Returns:
            ReasoningQuality assessment
        """
        issues = []
        
        # Validate naturalness (detect robotic phrases)
        naturalness_score, naturalness_issues = self._validate_naturalness(reasoning)
        issues.extend(naturalness_issues)
        
        # Validate accuracy (risk-reward ratios, technical accuracy)
        accuracy_score, accuracy_issues = self._validate_accuracy(reasoning)
        issues.extend(accuracy_issues)
        
        # Validate consistency (decision vs analysis alignment)
        consistency_score, consistency_issues = self._validate_consistency(reasoning)
        issues.extend(consistency_issues)
        
        # Measure diversity (content uniqueness)
        diversity_score = self._measure_diversity(reasoning)
        
        # Calculate overall score
        overall_score = np.mean([naturalness_score, diversity_score, accuracy_score, consistency_score])
        
        return ReasoningQuality(
            naturalness_score=naturalness_score,
            diversity_score=diversity_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            issues=issues,
            overall_score=overall_score
        )
    
    def _load_robotic_phrases(self) -> List[str]:
        """Load list of robotic phrases to detect."""
        return [
            "Of special significance is",
            "Particularly noteworthy is",
            "Especially important is",
            "Notably evident is",
            "Based on current analysis",
            "Technical evaluation suggests",
            "Market assessment indicates",
            "Current examination reveals",
            "particularly noteworthy",
            "especially significant",
            "of particular importance"
        ]
    
    def _load_consistency_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for consistency checking."""
        return {
            'bullish_decision': ['going long', 'long position', 'bullish', 'buy'],
            'bearish_decision': ['going short', 'short position', 'bearish', 'sell'],
            'hold_decision': ['staying hold', 'hold position', 'wait', 'patience'],
            'bullish_support': ['strong momentum', 'bullish alignment', 'upward', 'positive'],
            'bearish_support': ['weak momentum', 'bearish alignment', 'downward', 'negative'],
            'neutral_support': ['mixed signals', 'uncertain', 'balanced', 'consolidation']
        }
    
    def _validate_naturalness(self, reasoning: Dict[str, str]) -> Tuple[float, List[str]]:
        """Validate naturalness by detecting robotic phrases and broken structures."""
        issues = []
        total_phrases = 0
        robotic_count = 0
        
        for column, text in reasoning.items():
            if not text:
                continue
                
            # Count robotic phrases
            for phrase in self.robotic_phrases:
                if phrase.lower() in text.lower():
                    robotic_count += 1
                    issues.append(f"Robotic phrase '{phrase}' found in {column}")
            
            # Check for broken sentence structures
            if self._has_broken_sentences(text):
                issues.append(f"Broken sentence structure in {column}")
            
            # Count total phrases for scoring
            total_phrases += len(text.split('.'))
        
        # Calculate naturalness score (1.0 = no robotic phrases)
        if total_phrases > 0:
            naturalness_score = max(0.0, 1.0 - (robotic_count / total_phrases))
        else:
            naturalness_score = 1.0
        
        return naturalness_score, issues
    
    def _has_broken_sentences(self, text: str) -> bool:
        """Check for broken sentence structures."""
        # Pattern: number followed immediately by capital letter (e.g., "0.Particularly")
        if re.search(r'\d+\.[A-Z]', text):
            return True
        
        # Pattern: word followed immediately by capital letter (e.g., "word.Capital")
        if re.search(r'[a-z]\.[A-Z]', text):
            return True
        
        # Pattern: incomplete fragments
        if re.search(r'\.\s*\d+\s*\.', text):
            return True
        
        return False
    
    def _validate_accuracy(self, reasoning: Dict[str, str]) -> Tuple[float, List[str]]:
        """Validate technical accuracy, especially risk-reward ratios."""
        issues = []
        accuracy_score = 1.0
        
        # Check risk-reward ratios in execution_decision and risk_reward columns
        for column in ['execution_decision', 'risk_reward']:
            if column in reasoning:
                text = reasoning[column]
                ratio_issues = self._validate_risk_reward_ratios(text, column)
                issues.extend(ratio_issues)
                
                # Reduce accuracy score for each issue
                if ratio_issues:
                    accuracy_score -= 0.2 * len(ratio_issues)
        
        # Ensure score doesn't go below 0
        accuracy_score = max(0.0, accuracy_score)
        
        return accuracy_score, issues
    
    def _validate_risk_reward_ratios(self, text: str, column: str) -> List[str]:
        """Validate risk-reward ratios are realistic."""
        issues = []
        
        # Find risk-reward ratio patterns (e.g., "2.5:1", "0.0:1")
        ratio_pattern = r'(\d+\.?\d*):1'
        matches = re.findall(ratio_pattern, text)
        
        for ratio_str in matches:
            try:
                ratio = float(ratio_str)
                if ratio < 0.1:
                    issues.append(f"Unrealistic low risk-reward ratio {ratio}:1 in {column}")
                elif ratio > 5.0:
                    issues.append(f"Unrealistic high risk-reward ratio {ratio}:1 in {column}")
            except ValueError:
                issues.append(f"Invalid risk-reward ratio format in {column}")
        
        return issues
    
    def _validate_consistency(self, reasoning: Dict[str, str]) -> Tuple[float, List[str]]:
        """Validate consistency between decision and supporting analysis."""
        issues = []
        consistency_score = 1.0

        if 'decision' not in reasoning:
            return consistency_score, issues

        decision_text = reasoning['decision'].lower()

        # Determine decision direction
        decision_direction = self._extract_decision_direction(decision_text)

        if decision_direction == 'unknown':
            issues.append("Unable to determine decision direction")
            consistency_score -= 0.3
            return max(0.0, consistency_score), issues

        # CRITICAL: Check signal-decision alignment if signal data is available
        if hasattr(self, 'current_signal'):
            signal_direction = self._get_signal_direction(self.current_signal)
            if signal_direction != decision_direction:
                issues.append(f"CRITICAL: Decision text says '{decision_direction}' but signal indicates '{signal_direction}'")
                consistency_score -= 0.8  # Major penalty for signal misalignment

        # Check consistency with supporting columns (excluding risk_reward per user preferences)
        supporting_columns = ['pattern_recognition', 'context_analysis', 'psychology_assessment']

        for column in supporting_columns:
            if column in reasoning:
                support_direction = self._extract_support_direction(reasoning[column].lower())
                if support_direction != 'neutral' and support_direction != decision_direction:
                    issues.append(f"Decision direction ({decision_direction}) inconsistent with {column} analysis ({support_direction})")
                    consistency_score -= 0.2

        return max(0.0, consistency_score), issues
    
    def _extract_decision_direction(self, decision_text: str) -> str:
        """Extract decision direction from decision text."""
        if any(phrase in decision_text for phrase in self.consistency_patterns['bullish_decision']):
            return 'bullish'
        elif any(phrase in decision_text for phrase in self.consistency_patterns['bearish_decision']):
            return 'bearish'
        elif any(phrase in decision_text for phrase in self.consistency_patterns['hold_decision']):
            return 'hold'
        else:
            return 'unknown'
    
    def _extract_support_direction(self, support_text: str) -> str:
        """Extract support direction from analysis text."""
        bullish_count = sum(1 for phrase in self.consistency_patterns['bullish_support'] if phrase in support_text)
        bearish_count = sum(1 for phrase in self.consistency_patterns['bearish_support'] if phrase in support_text)
        neutral_count = sum(1 for phrase in self.consistency_patterns['neutral_support'] if phrase in support_text)
        
        if bullish_count > bearish_count and bullish_count > neutral_count:
            return 'bullish'
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            return 'bearish'
        else:
            return 'neutral'
    
    def _measure_diversity(self, reasoning: Dict[str, str]) -> float:
        """Measure content diversity to detect excessive repetition."""
        all_text = ' '.join(reasoning.values()).lower()
        
        # Split into phrases (sentences)
        phrases = [phrase.strip() for phrase in all_text.split('.') if phrase.strip()]
        
        if len(phrases) < 2:
            return 1.0
        
        # Count unique phrases
        unique_phrases = len(set(phrases))
        total_phrases = len(phrases)
        
        # Calculate diversity score
        diversity_score = unique_phrases / total_phrases
        
        return diversity_score
    
    def track_phrase_usage(self, reasoning: Dict[str, str]) -> None:
        """Track phrase usage across multiple reasoning generations."""
        for column, text in reasoning.items():
            phrases = [phrase.strip() for phrase in text.split('.') if phrase.strip()]
            for phrase in phrases:
                if phrase not in self.phrase_usage_tracker:
                    self.phrase_usage_tracker[phrase] = 0
                self.phrase_usage_tracker[phrase] += 1
    
    def _get_signal_direction(self, signal: int) -> str:
        """Convert signal value to direction string."""
        if signal == 1:
            return 'bullish'
        elif signal == 2:
            return 'bearish'
        else:
            return 'hold'

    def set_current_signal(self, signal: int) -> None:
        """Set current signal for validation."""
        self.current_signal = signal

    def get_overused_phrases(self, threshold: int = 3) -> List[Tuple[str, int]]:
        """Get phrases that are used more than the threshold."""
        return [(phrase, count) for phrase, count in self.phrase_usage_tracker.items()
                if count > threshold]
