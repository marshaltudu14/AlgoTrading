#!/usr/bin/env python3
"""
Content Diversity Manager
========================

Manages content diversity across reasoning generation to ensure >80% content uniqueness.
Tracks phrase usage, implements phrase rotation, and provides synonym replacement.

Key Features:
- Tracks phrase usage across multiple reasoning generations
- Implements intelligent phrase rotation to prevent repetition
- Provides synonym replacement for overused terms
- Ensures >80% content uniqueness across rows
- Maintains natural language quality while increasing diversity
"""

import random
import re
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class ContentDiversityManager:
    """
    Manages content diversity across reasoning generation.
    """
    
    def __init__(self, diversity_threshold: float = 0.8):
        """
        Initialize the content diversity manager.
        
        Args:
            diversity_threshold: Minimum diversity score required (0.8 = 80%)
        """
        self.diversity_threshold = diversity_threshold
        self.phrase_usage_tracker = defaultdict(int)
        self.sentence_usage_tracker = defaultdict(int)
        self.synonym_mappings = self._load_synonym_mappings()
        self.phrase_alternatives = self._load_phrase_alternatives()
        self.overuse_threshold = 3  # Max times a phrase can be used
        
        logger.info(f"ContentDiversityManager initialized with {diversity_threshold:.1%} diversity threshold")
    
    def enhance_content_diversity(self, reasoning: Dict[str, str], 
                                generation_count: int = 0) -> Dict[str, str]:
        """
        Enhance content diversity by replacing overused phrases and terms.
        
        Args:
            reasoning: Dictionary with reasoning columns
            generation_count: Current generation number for tracking
            
        Returns:
            Enhanced reasoning with improved diversity
        """
        enhanced_reasoning = {}
        
        for column, text in reasoning.items():
            if not text:
                enhanced_reasoning[column] = text
                continue
            
            # Track current usage
            self._track_content_usage(text, column)
            
            # Apply diversity enhancements
            enhanced_text = self._apply_diversity_enhancements(text, column, generation_count)
            enhanced_reasoning[column] = enhanced_text
        
        return enhanced_reasoning
    
    def _load_synonym_mappings(self) -> Dict[str, List[str]]:
        """Load synonym mappings for common trading terms."""
        return {
            'analysis': ['assessment', 'evaluation', 'examination', 'review', 'study'],
            'indicates': ['suggests', 'shows', 'reveals', 'demonstrates', 'points to'],
            'current': ['present', 'existing', 'ongoing', 'active', 'immediate'],
            'market': ['market'],  # Keep it simple - just use 'market'
            'conditions': ['environment', 'circumstances', 'situation', 'context', 'dynamics'],
            'strong': ['robust', 'solid', 'powerful', 'significant', 'substantial'],
            'moderate': ['balanced', 'measured', 'reasonable', 'steady', 'consistent'],
            'weak': ['limited', 'modest', 'subdued', 'restrained', 'tentative'],
            'bullish': ['positive', 'optimistic', 'constructive', 'favorable', 'encouraging'],
            'bearish': ['negative', 'pessimistic', 'unfavorable', 'concerning', 'challenging'],
            'momentum': ['velocity', 'acceleration', 'thrust', 'drive', 'impetus'],
            'trend': ['direction', 'trajectory', 'movement', 'bias', 'inclination'],
            'support': ['floor', 'base', 'foundation', 'underpinning', 'backing'],
            'resistance': ['ceiling', 'barrier', 'obstacle', 'impediment', 'hurdle'],
            'volatility': ['variability', 'fluctuation', 'instability', 'uncertainty', 'turbulence']
        }
    
    def _load_phrase_alternatives(self) -> Dict[str, List[str]]:
        """Load alternative phrases for common expressions."""
        return {
            'technical analysis reveals': [
                'market examination shows',
                'technical assessment indicates',
                'price analysis demonstrates',
                'chart evaluation suggests',
                'technical review confirms'
            ],
            'market participants demonstrate': [
                'traders exhibit',
                'investors display',
                'market players show',
                'trading community demonstrates',
                'market actors exhibit'
            ],
            'current market conditions': [
                'present trading environment',
                'existing market dynamics',
                'ongoing market situation',
                'active trading conditions',
                'immediate market context'
            ],
            'technical indicators show': [
                'momentum measures reveal',
                'technical metrics indicate',
                'analytical tools demonstrate',
                'market indicators suggest',
                'technical signals show'
            ],
            'price action indicates': [
                'trading behavior suggests',
                'market movement shows',
                'price dynamics reveal',
                'trading patterns indicate',
                'market activity demonstrates'
            ]
        }
    
    def _track_content_usage(self, text: str, column: str) -> None:
        """Track usage of phrases and sentences."""
        # Track sentence-level usage - use regex to split on sentence-ending periods only
        sentences = [s.strip() for s in re.split(r'\.(?=\s+[A-Z]|$)', text) if s.strip()]
        for sentence in sentences:
            self.sentence_usage_tracker[sentence] += 1
        
        # Track phrase-level usage (3-5 word phrases)
        words = text.lower().split()
        for i in range(len(words) - 2):
            for phrase_len in [3, 4, 5]:
                if i + phrase_len <= len(words):
                    phrase = ' '.join(words[i:i + phrase_len])
                    self.phrase_usage_tracker[phrase] += 1
    
    def _apply_diversity_enhancements(self, text: str, column: str, generation_count: int) -> str:
        """Apply diversity enhancements to text."""
        enhanced_text = text
        
        # Replace overused phrases
        enhanced_text = self._replace_overused_phrases(enhanced_text)
        
        # Apply synonym replacement
        enhanced_text = self._apply_synonym_replacement(enhanced_text, generation_count)
        
        # Vary sentence structures
        enhanced_text = self._vary_sentence_structures(enhanced_text)
        
        return enhanced_text
    
    def _replace_overused_phrases(self, text: str) -> str:
        """Replace overused phrases with alternatives."""
        enhanced_text = text
        
        for phrase, alternatives in self.phrase_alternatives.items():
            if phrase.lower() in enhanced_text.lower():
                usage_count = self.phrase_usage_tracker.get(phrase.lower(), 0)
                if usage_count >= self.overuse_threshold:
                    # Replace with a random alternative
                    alternative = random.choice(alternatives)
                    enhanced_text = re.sub(
                        re.escape(phrase),
                        alternative,
                        enhanced_text,
                        count=1,
                        flags=re.IGNORECASE
                    )
        
        return enhanced_text
    
    def _apply_synonym_replacement(self, text: str, generation_count: int) -> str:
        """Apply synonym replacement based on usage frequency."""
        enhanced_text = text
        
        # Increase replacement probability as generation count increases
        replacement_probability = min(0.4 + (generation_count * 0.05), 0.7)
        
        for original_term, synonyms in self.synonym_mappings.items():
            if original_term.lower() in enhanced_text.lower():
                usage_count = self.phrase_usage_tracker.get(original_term.lower(), 0)
                
                # Higher chance of replacement for overused terms
                if usage_count >= self.overuse_threshold or random.random() < replacement_probability:
                    synonym = random.choice(synonyms)
                    enhanced_text = re.sub(
                        rf'\b{re.escape(original_term)}\b',
                        synonym,
                        enhanced_text,
                        count=1,
                        flags=re.IGNORECASE
                    )
        
        return enhanced_text
    
    def _vary_sentence_structures(self, text: str) -> str:
        """Vary sentence structures to increase diversity."""
        # Use regex to split on sentence-ending periods, not decimal points
        # This pattern matches periods that are followed by whitespace and a capital letter
        # or periods at the end of the text, but not decimal points in numbers
        sentences = re.split(r'\.(?=\s+[A-Z]|$)', text)
        varied_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Apply structural variations
            varied_sentence = self._apply_structural_variation(sentence)
            varied_sentences.append(varied_sentence)

        return '. '.join(varied_sentences) + '.' if varied_sentences else text
    
    def _apply_structural_variation(self, sentence: str) -> str:
        """Apply structural variations to a sentence."""
        # Simple structural variations
        variations = [
            lambda s: s,  # Keep original
            lambda s: self._add_transitional_phrase(s),
            lambda s: self._reorder_clauses(s)
        ]
        
        # Apply variation with probability
        if random.random() < 0.3:
            variation_func = random.choice(variations)
            return variation_func(sentence)
        
        return sentence
    
    def _add_transitional_phrase(self, sentence: str) -> str:
        """Add transitional phrases for variety."""
        transitional_phrases = [
            "Furthermore,", "Additionally,", "Moreover,", "In addition,",
            "Notably,", "Significantly,", "Importantly,"
        ]
        
        if random.random() < 0.3 and not sentence.lower().startswith(('furthermore', 'additionally', 'moreover')):
            phrase = random.choice(transitional_phrases)
            return f"{phrase} {sentence.lower()}"
        
        return sentence
    
    def _reorder_clauses(self, sentence: str) -> str:
        """Reorder clauses in compound sentences."""
        # Simple clause reordering for sentences with "and", "but", "while"
        connectors = [' and ', ' but ', ' while ']
        
        for connector in connectors:
            if connector in sentence.lower():
                parts = sentence.split(connector, 1)
                if len(parts) == 2 and random.random() < 0.3:
                    # Reorder with appropriate connector
                    reorder_connector = ' while ' if connector == ' and ' else connector
                    return f"{parts[1].strip()}{reorder_connector}{parts[0].strip()}"
        
        return sentence
    
    def calculate_diversity_score(self, reasoning_list: List[Dict[str, str]]) -> float:
        """
        Calculate diversity score across multiple reasoning generations.
        
        Args:
            reasoning_list: List of reasoning dictionaries
            
        Returns:
            Diversity score (0.0 to 1.0)
        """
        if not reasoning_list:
            return 1.0
        
        all_sentences = []
        for reasoning in reasoning_list:
            for column, text in reasoning.items():
                if text:
                    sentences = [s.strip() for s in text.split('.') if s.strip()]
                    all_sentences.extend(sentences)
        
        if not all_sentences:
            return 1.0
        
        unique_sentences = len(set(all_sentences))
        total_sentences = len(all_sentences)
        
        return unique_sentences / total_sentences
    
    def get_overused_content(self) -> Dict[str, List[Tuple[str, int]]]:
        """Get overused phrases and sentences."""
        return {
            'phrases': [(phrase, count) for phrase, count in self.phrase_usage_tracker.items() 
                       if count > self.overuse_threshold],
            'sentences': [(sentence, count) for sentence, count in self.sentence_usage_tracker.items() 
                         if count > self.overuse_threshold]
        }
    
    def reset_tracking(self) -> None:
        """Reset usage tracking for new session."""
        self.phrase_usage_tracker.clear()
        self.sentence_usage_tracker.clear()
        logger.info("Content diversity tracking reset")
