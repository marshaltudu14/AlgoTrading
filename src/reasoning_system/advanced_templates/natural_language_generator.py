#!/usr/bin/env python3
"""
Natural Language Generator for Enhanced Reasoning System
=======================================================

Advanced rule-based natural language generation for sophisticated reasoning text.
Maintains technical accuracy while adding natural variety and professional language.

Key Features:
- Sophisticated rule-based natural language patterns
- Multiple variations for same concepts (5-10 ways to express each idea)
- Contextual phrase selection based on market strength/weakness
- Technical accuracy preservation with natural variety
- Fast processing without LLM overhead
"""

import random
import re
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NaturalLanguageGenerator:
    """
    Natural language generator for sophisticated reasoning text.
    """
    
    def __init__(self):
        """Initialize natural language generator."""
        self.sentence_starters = self._load_sentence_starters()
        self.transition_phrases = self._load_transition_phrases()
        self.technical_synonyms = self._load_technical_synonyms()
        self.market_descriptors = self._load_market_descriptors()
        self.professional_phrases = self._load_professional_phrases()
        
        logger.info("NaturalLanguageGenerator initialized with natural language patterns")
    
    def enhance_reasoning_text(self, text: str, reasoning_type: str = 'general',
                             market_strength: float = 0.5, confidence_level: float = 0.5) -> str:
        """
        Enhance reasoning text with focus on technical accuracy over language flow.

        Args:
            text: Base reasoning text to enhance
            reasoning_type: Type of reasoning (pattern, context, psychology, etc.)
            market_strength: Market strength (0-1) for intensity selection
            confidence_level: Confidence level (0-1) for certainty expressions

        Returns:
            Enhanced reasoning text focused on accuracy and specificity
        """
        if not text or not text.strip():
            return self._get_default_text(reasoning_type)

        try:
            # Apply minimal enhancements focused on technical accuracy
            enhanced_text = self._apply_technical_synonyms(text)
            enhanced_text = self._apply_market_descriptors(enhanced_text, market_strength)
            # Removed excessive transition improvements and natural flow for better accuracy
            enhanced_text = self._ensure_technical_precision(enhanced_text)

            return enhanced_text

        except Exception as e:
            logger.error(f"Error enhancing text: {str(e)}")
            return text  # Return original text if enhancement fails
    
    def generate_natural_sentence(self, concept: str, context: Dict[str, Any] = None) -> str:
        """Generate natural sentence for a given concept."""
        context = context or {}
        
        # Get base sentence structure
        sentence_starter = self._get_contextual_starter(concept, context)
        
        # Add concept-specific content
        concept_content = self._generate_concept_content(concept, context)
        
        # Add professional conclusion
        conclusion = self._get_professional_conclusion(concept, context)
        
        # Combine with natural transitions
        sentence = f"{sentence_starter} {concept_content}"
        if conclusion:
            transition = random.choice(self.transition_phrases['conclusion'])
            sentence += f" {transition} {conclusion}"
        
        return sentence
    
    def _load_sentence_starters(self) -> Dict[str, List[str]]:
        """Load sentence starter variations."""
        return {
            'analysis': [
                "Price action reveals",
                "Market dynamics show",
                "Trading patterns indicate",
                "Chart analysis demonstrates",
                "Market structure suggests",
                "Price behavior confirms",
                "Technical setup reveals",
                "Market conditions show"
            ],
            'pattern': [
                "Pattern recognition identifies",
                "Formation analysis reveals",
                "Chart pattern examination shows",
                "Technical pattern assessment indicates",
                "Pattern development suggests",
                "Formation characteristics demonstrate",
                "Pattern analysis confirms",
                "Structure examination reveals"
            ],
            'momentum': [
                "Momentum analysis indicates",
                "Momentum assessment reveals",
                "Momentum characteristics suggest",
                "Momentum evaluation shows",
                "Momentum dynamics demonstrate",
                "Momentum indicators confirm",
                "Momentum patterns reveal",
                "Momentum behavior indicates"
            ],
            'trend': [
                "Trend analysis demonstrates",
                "Trend assessment indicates",
                "Trend characteristics reveal",
                "Trend evaluation suggests",
                "Trend dynamics show",
                "Trend behavior confirms",
                "Trend development indicates",
                "Trend structure demonstrates"
            ],
            'risk': [
                "Risk assessment indicates",
                "Risk analysis reveals",
                "Risk evaluation demonstrates",
                "Risk characteristics suggest",
                "Risk dynamics show",
                "Risk factors indicate",
                "Risk profile reveals",
                "Risk assessment confirms"
            ]
        }
    
    def _load_transition_phrases(self) -> Dict[str, List[str]]:
        """Load transition phrase variations."""
        return {
            'continuation': [
                "while simultaneously",
                "concurrently",
                "at the same time",
                "in parallel",
                "alongside this",
                "correspondingly",
                "in conjunction with",
                "complementing this"
            ],
            'contrast': [
                "however",
                "nevertheless",
                "conversely",
                "in contrast",
                "on the other hand",
                "alternatively",
                "despite this",
                "notwithstanding"
            ],
            'emphasis': [
                "notably",
                "remarkably",
                "significantly",
                "crucially",
                "fundamentally",
                "clearly",
                "evidently",
                "substantially"
            ],
            'conclusion': [
                "suggesting",
                "indicating",
                "implying",
                "demonstrating",
                "confirming",
                "revealing",
                "establishing",
                "supporting"
            ],
            'causation': [
                "consequently",
                "as a result",
                "therefore",
                "thus",
                "accordingly",
                "hence",
                "subsequently",
                "resulting in"
            ]
        }
    
    def _load_technical_synonyms(self) -> Dict[str, List[str]]:
        """Load technical term synonyms for variety."""
        return {
            'price': ['price'],  # Keep it simple - just use 'price'
            'trend': ['trend', 'direction', 'trajectory', 'movement', 'bias', 'inclination'],
            'momentum': ['momentum', 'velocity', 'acceleration', 'thrust', 'impetus', 'drive'],
            'volatility': ['volatility', 'variability', 'fluctuation', 'instability', 'uncertainty'],
            'support': ['support', 'floor', 'base', 'foundation', 'underpinning'],
            'resistance': ['resistance', 'ceiling', 'barrier', 'obstacle', 'impediment'],
            'breakout': ['breakout', 'breach', 'penetration', 'breakthrough', 'escape'],
            'consolidation': ['consolidation', 'range', 'sideways movement', 'lateral trading', 'equilibrium'],
            'reversal': ['reversal', 'turnaround', 'change', 'shift', 'transition'],
            'continuation': ['continuation', 'persistence', 'maintenance', 'extension', 'prolongation']
        }
    
    def _load_market_descriptors(self) -> Dict[str, Dict[str, List[str]]]:
        """Load market strength descriptors."""
        return {
            'strength': {
                'weak': ['modest', 'limited', 'restrained', 'subdued', 'tentative'],
                'moderate': ['moderate', 'balanced', 'measured', 'steady', 'consistent'],
                'strong': ['pronounced', 'significant', 'notable', 'substantial', 'compelling'],
                'very_strong': ['exceptional', 'remarkable', 'outstanding', 'extraordinary', 'powerful']
            },
            'direction': {
                'bullish': ['positive', 'constructive', 'favorable', 'optimistic', 'encouraging'],
                'bearish': ['negative', 'unfavorable', 'pessimistic', 'concerning', 'challenging'],
                'neutral': ['balanced', 'equilibrium', 'stable', 'measured', 'even']
            },
            'certainty': {
                'high': ['definitive', 'clear', 'unambiguous', 'decisive', 'conclusive'],
                'moderate': ['reasonable', 'adequate', 'sufficient', 'acceptable', 'probable'],
                'low': ['tentative', 'uncertain', 'questionable', 'preliminary', 'inconclusive']
            }
        }
    
    def _load_professional_phrases(self) -> Dict[str, List[str]]:
        """Load professional language phrases."""
        return {
            'assessment': [
                "comprehensive evaluation",
                "detailed assessment",
                "thorough analysis",
                "systematic examination",
                "methodical review"
            ],
            'indication': [
                "suggests the possibility",
                "indicates the potential",
                "points toward",
                "implies the likelihood",
                "demonstrates the probability"
            ],
            'confirmation': [
                "validates the assessment",
                "corroborates the analysis",
                "substantiates the evaluation",
                "reinforces the conclusion",
                "supports the interpretation"
            ],
            'caution': [
                "warrants careful consideration",
                "requires prudent evaluation",
                "demands cautious assessment",
                "necessitates measured approach",
                "calls for selective positioning"
            ]
        }
    
    def _apply_sentence_variations(self, text: str) -> str:
        """Apply sentence structure variations."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        enhanced_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Apply variations to sentence structure
            enhanced_sentence = self._vary_sentence_structure(sentence)
            enhanced_sentences.append(enhanced_sentence)
        
        return '. '.join(enhanced_sentences) + '.'
    
    def _vary_sentence_structure(self, sentence: str) -> str:
        """Vary individual sentence structure."""
        # Simple sentence structure variations
        variations = [
            lambda s: s,  # Keep original
            lambda s: self._add_professional_qualifier(s),
            lambda s: self._restructure_with_emphasis(s),
            lambda s: self._add_contextual_phrase(s)
        ]
        
        variation_func = random.choice(variations)
        return variation_func(sentence)
    
    def _add_professional_qualifier(self, sentence: str) -> str:
        """Add professional qualifier to sentence (reduced frequency to prevent overuse)."""
        qualifiers = [
            "Technical analysis shows",
            "Market conditions indicate",
            "Current data suggests"
        ]

        if random.random() < 0.1:  # Reduced to 10% chance to prevent overuse
            qualifier = random.choice(qualifiers)
            return f"{qualifier} {sentence.lower()}"

        return sentence
    
    def _restructure_with_emphasis(self, sentence: str) -> str:
        """Restructure sentence with natural emphasis (removed robotic phrases)."""
        # REMOVED: Robotic emphasis patterns that cause quality issues
        # Instead, use natural sentence flow without artificial restructuring

        # Apply natural emphasis through word choice rather than sentence restructuring
        if random.random() < 0.15:  # Reduced probability and use natural enhancement
            return self._add_natural_emphasis(sentence)

        return sentence

    def _add_natural_emphasis(self, sentence: str) -> str:
        """Add natural emphasis through word choice rather than sentence restructuring."""
        # Natural emphasis words that can be inserted naturally
        natural_emphasis = [
            "clearly", "evidently", "notably", "significantly",
            "importantly", "remarkably", "substantially"
        ]

        # Minimal emphasis for technical accuracy - focus on data not language
        return sentence

    def _ensure_technical_precision(self, text: str) -> str:
        """Ensure technical precision and remove excessive transitions."""
        # Remove excessive transition words that make text robotic
        excessive_transitions = [
            "Additionally, ", "Furthermore, ", "Significantly, ", "Moreover, ",
            "Importantly, ", "Notably, ", "In addition, ", "Current data suggests ",
            "Technical comprehensive evaluation shows ", "immediate data suggests ",
            "present data suggests ", "ongoing data suggests ", "existing data suggests ",
            "active data suggests "
        ]

        for transition in excessive_transitions:
            text = text.replace(transition, "")

        # Fix decimal number formatting issues
        # Pattern: "48600. 2" -> "48600.2", "50. 0" -> "50.0"
        text = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', text)

        # Fix "at X. Y" patterns for indicators
        # Pattern: "RSI at 50. 0" -> "RSI at 50.0"
        text = re.sub(r'(at \d+)\.\s+(\d+)', r'\1.\2', text)

        # Fix percentage patterns
        # Pattern: "0. 1%" -> "0.1%"
        text = re.sub(r'(\d+)\.\s+(\d+%)', r'\1.\2', text)

        # Clean up double spaces and ensure proper formatting
        text = " ".join(text.split())

        return text
    
    def _add_contextual_phrase(self, sentence: str) -> str:
        """Add contextual phrase to sentence (reduced frequency to prevent overuse)."""
        contextual_phrases = [
            "given current conditions",
            "based on technical factors"
        ]

        if random.random() < 0.05:  # Reduced to 5% chance to prevent overuse
            context = random.choice(contextual_phrases)
            return f"{sentence}, {context}"

        return sentence

    def _apply_technical_synonyms(self, text: str) -> str:
        """Apply technical synonym variations."""
        enhanced_text = text

        # Apply synonyms with probability to maintain some original terms
        for original_term, synonyms in self.technical_synonyms.items():
            if original_term in enhanced_text.lower():
                if random.random() < 0.4:  # 40% chance to replace
                    synonym = random.choice(synonyms)
                    # Replace first occurrence only to maintain readability
                    enhanced_text = re.sub(
                        rf'\b{re.escape(original_term)}\b',
                        synonym,
                        enhanced_text,
                        count=1,
                        flags=re.IGNORECASE
                    )

        return enhanced_text

    def _apply_market_descriptors(self, text: str, market_strength: float) -> str:
        """Apply market strength descriptors."""
        # Determine strength category
        if market_strength > 0.8:
            strength_category = 'very_strong'
        elif market_strength > 0.6:
            strength_category = 'strong'
        elif market_strength > 0.4:
            strength_category = 'moderate'
        else:
            strength_category = 'weak'

        # Replace generic strength terms with specific descriptors
        strength_terms = ['strong', 'moderate', 'weak', 'significant', 'notable']
        for term in strength_terms:
            if term in text.lower():
                if random.random() < 0.5:  # 50% chance to enhance
                    descriptors = self.market_descriptors['strength'][strength_category]
                    replacement = random.choice(descriptors)
                    text = re.sub(
                        rf'\b{re.escape(term)}\b',
                        replacement,
                        text,
                        count=1,
                        flags=re.IGNORECASE
                    )

        return text

    def _apply_professional_language(self, text: str, confidence_level: float) -> str:
        """Apply professional language enhancements."""
        # Determine confidence category
        if confidence_level > 0.7:
            confidence_category = 'high'
        elif confidence_level > 0.5:
            confidence_category = 'moderate'
        else:
            confidence_category = 'low'

        # Enhance with professional phrases
        if random.random() < 0.3:  # 30% chance to add professional language
            professional_phrase = random.choice(self.professional_phrases['assessment'])
            text = text.replace('analysis', professional_phrase, 1)

        return text

    def _apply_transition_improvements(self, text: str) -> str:
        """Apply transition phrase improvements."""
        # Improve basic transitions with more sophisticated ones
        basic_transitions = {
            'and': self.transition_phrases['continuation'],
            'but': self.transition_phrases['contrast'],
            'so': self.transition_phrases['causation'],
            'also': self.transition_phrases['continuation']
        }

        for basic, sophisticated in basic_transitions.items():
            if f' {basic} ' in text.lower():
                if random.random() < 0.4:  # 40% chance to improve
                    replacement = random.choice(sophisticated)
                    text = re.sub(
                        rf'\b{re.escape(basic)}\b',
                        replacement,
                        text,
                        count=1,
                        flags=re.IGNORECASE
                    )

        return text

    def _ensure_natural_flow(self, text: str) -> str:
        """Ensure natural flow and readability."""
        # CRITICAL: Fix broken sentence structures first
        text = self._fix_sentence_fragments(text)

        # Remove redundant phrases
        text = re.sub(r'\b(very very|really really|quite quite)\b', r'\1', text, flags=re.IGNORECASE)

        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.!?])', r'\1', text)

        # Ensure proper capitalization
        sentences = re.split(r'([.!?]+)', text)
        for i in range(0, len(sentences), 2):
            if sentences[i].strip():
                sentences[i] = sentences[i].strip().capitalize()

        return ''.join(sentences)

    def _fix_sentence_fragments(self, text: str) -> str:
        """Fix broken sentence structures and number formatting issues."""
        # CRITICAL: Fix decimal numbers being split incorrectly
        # Pattern: "48600. 2" -> "48600.2", "50. 0" -> "50.0"
        text = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', text)

        # CRITICAL: Fix patterns where sentences are broken by missing spaces after periods
        # Pattern: "word.CapitalLetter" -> "word. CapitalLetter"
        text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', text)

        # CRITICAL: Fix fragments like "score of 0.Of special significance"
        # Pattern: "score of X.SomePhrase" -> "score of X. SomePhrase"
        text = re.sub(r'(score of \d+(?:\.\d+)?)\.([A-Z])', r'\1. \2', text)

        # CRITICAL: Fix fragments like "ratio 2.5:1.Technical"
        # Pattern: "ratio X:Y.SomePhrase" -> "ratio X:Y. SomePhrase"
        text = re.sub(r'(ratio \d+(?:\.\d+)?:\d+)\.([A-Z])', r'\1. \2', text)

        # CRITICAL: Fix "at X. Y" patterns for indicators
        # Pattern: "RSI at 50. 0" -> "RSI at 50.0"
        text = re.sub(r'(at \d+)\.\s+(\d+)', r'\1.\2', text)

        # CRITICAL: Fix percentage patterns
        # Pattern: "0. 1%" -> "0.1%"
        text = re.sub(r'(\d+)\.\s+(\d+%)', r'\1.\2', text)

        # Fix indicator name capitalization and formatting
        text = re.sub(r'\bRsi\b', 'RSI', text)
        text = re.sub(r'\bMacd\b', 'MACD', text)
        text = re.sub(r'\bEma-(\d+)\b', r'EMA-\1', text)
        text = re.sub(r'\bSma-(\d+)\b', r'SMA-\1', text)
        text = re.sub(r'\bAtr\b', 'ATR', text)
        text = re.sub(r'\bAdx\b', 'ADX', text)
        text = re.sub(r'\bCci\b', 'CCI', text)

        # Fix awkward phrases
        text = re.sub(r'\bimpetus\b', 'momentum', text)
        text = re.sub(r'\bthrust\b', 'momentum', text)
        text = re.sub(r'\bvelocity\b', 'momentum', text)
        text = re.sub(r'\bdrive\b', 'momentum', text)
        text = re.sub(r'\bacceleration\b', 'momentum', text)
        text = re.sub(r'\bfloor levels\b', 'support levels', text)
        text = re.sub(r'\bceiling levels\b', 'resistance levels', text)
        text = re.sub(r'\bbarrier\b', 'resistance', text)
        text = re.sub(r'\bobstacle\b', 'resistance', text)
        text = re.sub(r'\bimpediment\b', 'resistance', text)
        text = re.sub(r'\bhurdle\b', 'resistance', text)
        text = re.sub(r'\bfoundation\b', 'support', text)
        text = re.sub(r'\bunderpinning\b', 'support', text)
        text = re.sub(r'\bbacking\b', 'support', text)
        text = re.sub(r'\bbase\b', 'support', text)

        # Fix redundant phrases
        text = re.sub(r'\bHowever, rsi and macd showing divergent momentum signals\. However, rsi and macd showing divergent momentum signals\b',
                     'However, RSI and MACD showing divergent momentum signals', text)
        text = re.sub(r'\bgiven current conditions\b', '', text)
        text = re.sub(r'\bgiven ongoing context\b', '', text)
        text = re.sub(r'\bgiven present dynamics\b', '', text)
        text = re.sub(r'\bbased on technical factors\b', '', text)
        text = re.sub(r'\bTechnical analysis shows\b', '', text)
        text = re.sub(r'\bMarket conditions indicate\b', '', text)

        # Clean up multiple consecutive periods and spaces
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+', ' ', text)

        # Ensure sentences end properly
        text = re.sub(r'([^.!?])\s*$', r'\1.', text)

        return text.strip()

    def _get_contextual_starter(self, concept: str, context: Dict[str, Any]) -> str:
        """Get contextual sentence starter."""
        concept_starters = {
            'pattern': self.sentence_starters['pattern'],
            'momentum': self.sentence_starters['momentum'],
            'trend': self.sentence_starters['trend'],
            'risk': self.sentence_starters['risk']
        }

        starters = concept_starters.get(concept, self.sentence_starters['analysis'])
        return random.choice(starters)

    def _generate_concept_content(self, concept: str, context: Dict[str, Any]) -> str:
        """Generate concept-specific content."""
        concept_content = {
            'bullish_momentum': "upward momentum characteristics with positive technical indicators",
            'bearish_momentum': "downward momentum characteristics with negative technical indicators",
            'trend_continuation': "trend continuation patterns with sustained directional bias",
            'consolidation': "consolidation characteristics with range-bound price behavior",
            'reversal_potential': "reversal potential with momentum shift indicators"
        }

        return concept_content.get(concept, "current market conditions with technical analysis")

    def _get_professional_conclusion(self, concept: str, context: Dict[str, Any]) -> str:
        """Get professional conclusion phrase."""
        conclusions = [
            "supporting the current assessment",
            "reinforcing the technical outlook",
            "validating the market analysis",
            "confirming the directional bias",
            "substantiating the technical evaluation"
        ]

        if random.random() < 0.6:  # 60% chance to add conclusion
            return random.choice(conclusions)

        return ""

    def _get_default_text(self, reasoning_type: str) -> str:
        """Get default text when enhancement fails."""
        defaults = {
            'pattern': "Technical pattern analysis indicates current market conditions with balanced characteristics.",
            'context': "Market context analysis reveals standard trading environment with typical dynamics.",
            'psychology': "Market psychology assessment shows balanced participant sentiment with measured behavior.",
            'execution': "Execution analysis suggests cautious positioning with selective market participation.",
            'risk': "Risk assessment indicates moderate conditions with standard risk-reward parameters."
        }

        return defaults.get(reasoning_type, "Technical analysis indicates current market conditions with balanced characteristics.")
