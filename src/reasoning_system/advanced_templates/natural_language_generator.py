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
        Enhance reasoning text with natural language variations.
        
        Args:
            text: Base reasoning text to enhance
            reasoning_type: Type of reasoning (pattern, context, psychology, etc.)
            market_strength: Market strength (0-1) for intensity selection
            confidence_level: Confidence level (0-1) for certainty expressions
            
        Returns:
            Enhanced natural language reasoning text
        """
        if not text or not text.strip():
            return self._get_default_text(reasoning_type)
        
        try:
            # Apply natural language enhancements
            enhanced_text = self._apply_sentence_variations(text)
            enhanced_text = self._apply_technical_synonyms(enhanced_text)
            enhanced_text = self._apply_market_descriptors(enhanced_text, market_strength)
            enhanced_text = self._apply_professional_language(enhanced_text, confidence_level)
            enhanced_text = self._apply_transition_improvements(enhanced_text)
            enhanced_text = self._ensure_natural_flow(enhanced_text)
            
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
                "Technical analysis reveals",
                "Market analysis indicates",
                "Current assessment shows",
                "Detailed examination demonstrates",
                "Comprehensive analysis suggests",
                "Technical evaluation confirms",
                "Market examination reveals",
                "Current analysis indicates"
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
                "particularly noteworthy",
                "especially significant",
                "of particular importance",
                "notably",
                "remarkably",
                "significantly",
                "crucially",
                "fundamentally"
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
            'price': ['price', 'market value', 'trading level', 'quotation', 'market price'],
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
        """Add professional qualifier to sentence."""
        qualifiers = [
            "Based on current analysis,",
            "Technical evaluation suggests",
            "Market assessment indicates",
            "Current examination reveals"
        ]
        
        if random.random() < 0.3:  # 30% chance to add qualifier
            qualifier = random.choice(qualifiers)
            return f"{qualifier} {sentence.lower()}"
        
        return sentence
    
    def _restructure_with_emphasis(self, sentence: str) -> str:
        """Restructure sentence with emphasis."""
        emphasis_patterns = [
            "Particularly noteworthy is",
            "Of special significance is",
            "Especially important is",
            "Notably evident is"
        ]
        
        if random.random() < 0.2:  # 20% chance to restructure
            emphasis = random.choice(emphasis_patterns)
            return f"{emphasis} {sentence.lower()}"
        
        return sentence
    
    def _add_contextual_phrase(self, sentence: str) -> str:
        """Add contextual phrase to sentence."""
        contextual_phrases = [
            "within the current market environment",
            "given present market conditions",
            "considering current technical factors",
            "in the context of recent market behavior"
        ]
        
        if random.random() < 0.25:  # 25% chance to add context
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
