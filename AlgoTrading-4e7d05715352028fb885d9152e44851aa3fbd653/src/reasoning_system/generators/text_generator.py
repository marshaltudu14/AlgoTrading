#!/usr/bin/env python3
"""
Text Generator
=============

Handles professional trading language generation, template management,
and text quality enhancement for reasoning output.
"""

import re
import random
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TextGenerator:
    """
    Generates and enhances professional trading language for reasoning text.
    
    Provides utilities for text formatting, professional language enhancement,
    and consistency validation across reasoning components.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the text generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Load professional language components
        self.professional_terms = self._load_professional_terms()
        self.transition_phrases = self._load_transition_phrases()
        self.confidence_expressions = self._load_confidence_expressions()
        
        # Configuration
        self.config.setdefault('min_reasoning_length', 50)
        self.config.setdefault('max_reasoning_length', 300)
        self.config.setdefault('use_professional_enhancement', True)
        
        logger.info("TextGenerator initialized with professional language components")
    
    def enhance_reasoning_text(self, text: str, reasoning_type: str = 'general') -> str:
        """
        Enhance reasoning text with professional language and formatting.
        
        Args:
            text: Raw reasoning text to enhance
            reasoning_type: Type of reasoning (pattern, context, psychology, etc.)
            
        Returns:
            Enhanced professional reasoning text
        """
        if not text or not text.strip():
            return self._get_default_text(reasoning_type)
        
        try:
            # Clean and format text
            enhanced_text = self._clean_text(text)
            
            # Apply professional language enhancement
            if self.config['use_professional_enhancement']:
                enhanced_text = self._apply_professional_language(enhanced_text, reasoning_type)
            
            # Ensure proper sentence structure
            enhanced_text = self._ensure_sentence_structure(enhanced_text)
            
            # Validate length constraints
            enhanced_text = self._validate_length(enhanced_text, reasoning_type)
            
            return enhanced_text
            
        except Exception as e:
            logger.error(f"Error enhancing text: {str(e)}")
            return text  # Return original text if enhancement fails
    
    def generate_confidence_expression(self, confidence_score: int) -> str:
        """
        Generate appropriate confidence expression based on score.
        
        Args:
            confidence_score: Confidence score (0-100)
            
        Returns:
            Professional confidence expression
        """
        if confidence_score >= 85:
            expressions = self.confidence_expressions['high']
        elif confidence_score >= 65:
            expressions = self.confidence_expressions['moderate']
        elif confidence_score >= 45:
            expressions = self.confidence_expressions['low']
        else:
            expressions = self.confidence_expressions['very_low']
        
        return random.choice(expressions)
    
    def format_percentage(self, value: float, context: str = 'general') -> str:
        """
        Format percentage values with appropriate professional language.
        
        Args:
            value: Percentage value to format
            context: Context for formatting (distance, change, etc.)
            
        Returns:
            Professionally formatted percentage string
        """
        if abs(value) < 0.1:
            return "minimal"
        elif abs(value) < 0.5:
            return "slight"
        elif abs(value) < 1.0:
            return "moderate"
        elif abs(value) < 2.0:
            return "significant"
        else:
            return "substantial"
    
    def format_relative_level(self, distance: float, level_type: str = 'support') -> str:
        """
        Format distance to levels with relative language.
        
        Args:
            distance: Distance percentage
            level_type: Type of level (support, resistance, etc.)
            
        Returns:
            Relative level description
        """
        if distance < 0.5:
            proximity = "immediate proximity to"
        elif distance < 1.0:
            proximity = "close proximity to"
        elif distance < 2.0:
            proximity = "moderate distance from"
        else:
            proximity = "significant distance from"
        
        return f"{proximity} key {level_type} zone"
    
    def _load_professional_terms(self) -> Dict[str, List[str]]:
        """Load professional trading terminology."""
        return {
            'trend_descriptors': [
                'directional bias', 'momentum characteristics', 'trend dynamics',
                'price trajectory', 'market direction', 'trending behavior'
            ],
            'strength_descriptors': [
                'robust', 'solid', 'moderate', 'weak', 'marginal', 'substantial'
            ],
            'market_conditions': [
                'market environment', 'trading conditions', 'market structure',
                'price action', 'market dynamics', 'trading landscape'
            ],
            'technical_terms': [
                'confluence', 'divergence', 'momentum', 'volatility',
                'support', 'resistance', 'breakout', 'consolidation'
            ],
            'action_terms': [
                'suggests', 'indicates', 'demonstrates', 'reveals',
                'presents', 'exhibits', 'displays', 'shows'
            ]
        }
    
    def _load_transition_phrases(self) -> Dict[str, List[str]]:
        """Load transition phrases for connecting ideas."""
        return {
            'continuation': [
                'furthermore', 'additionally', 'moreover', 'in addition',
                'building on this', 'extending the analysis'
            ],
            'contrast': [
                'however', 'conversely', 'on the other hand', 'alternatively',
                'in contrast', 'nevertheless'
            ],
            'causation': [
                'consequently', 'as a result', 'therefore', 'thus',
                'leading to', 'resulting in'
            ],
            'emphasis': [
                'notably', 'particularly', 'especially', 'significantly',
                'importantly', 'crucially'
            ]
        }
    
    def _load_confidence_expressions(self) -> Dict[str, List[str]]:
        """Load confidence expression templates."""
        return {
            'high': [
                'strong conviction', 'high confidence', 'clear indication',
                'definitive signal', 'robust evidence', 'compelling setup'
            ],
            'moderate': [
                'moderate confidence', 'reasonable indication', 'solid evidence',
                'balanced assessment', 'measured conviction', 'prudent evaluation'
            ],
            'low': [
                'cautious assessment', 'limited conviction', 'tentative indication',
                'preliminary signal', 'guarded optimism', 'reserved confidence'
            ],
            'very_low': [
                'minimal conviction', 'uncertain indication', 'unclear signal',
                'ambiguous evidence', 'limited clarity', 'inconclusive assessment'
            ]
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text formatting."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Ensure proper sentence endings
        if not text.endswith('.'):
            text += '.'
        
        # Fix common formatting issues
        text = re.sub(r'\s+\.', '.', text)  # Remove space before period
        text = re.sub(r'\.+', '.', text)    # Remove multiple periods
        text = re.sub(r'\s*,\s*', ', ', text)  # Normalize comma spacing
        
        return text
    
    def _apply_professional_language(self, text: str, reasoning_type: str) -> str:
        """Apply professional language enhancements."""
        # This is a simplified version - in a full implementation,
        # you might use more sophisticated NLP techniques
        
        # Replace basic terms with professional equivalents
        replacements = {
            'goes up': 'advances',
            'goes down': 'declines',
            'big': 'significant',
            'small': 'minimal',
            'good': 'favorable',
            'bad': 'unfavorable',
            'shows': 'demonstrates',
            'tells us': 'indicates',
            'looks like': 'suggests'
        }
        
        for basic, professional in replacements.items():
            text = re.sub(r'\b' + re.escape(basic) + r'\b', professional, text, flags=re.IGNORECASE)
        
        return text
    
    def _ensure_sentence_structure(self, text: str) -> str:
        """Ensure proper sentence structure and flow."""
        # Use regex to split on sentence-ending periods, not decimal points
        # This pattern matches periods that are followed by whitespace and a capital letter
        # or periods at the end of the text, but not decimal points in numbers
        sentences = re.split(r'\.(?=\s+[A-Z]|$)', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Capitalize first letter of each sentence
        formatted_sentences = []
        for sentence in sentences:
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                formatted_sentences.append(sentence)

        return '. '.join(formatted_sentences) + '.'
    
    def _validate_length(self, text: str, reasoning_type: str) -> str:
        """Validate and adjust text length."""
        min_length = self.config['min_reasoning_length']
        max_length = self.config['max_reasoning_length']
        
        if len(text) < min_length:
            # Text too short - add professional filler
            text = self._extend_text(text, reasoning_type)
        elif len(text) > max_length:
            # Text too long - trim while maintaining meaning
            text = self._trim_text(text, max_length)
        
        return text
    
    def _extend_text(self, text: str, reasoning_type: str) -> str:
        """Extend text with relevant professional content."""
        extensions = {
            'pattern': "This pattern formation requires careful monitoring for confirmation.",
            'context': "Market context analysis supports this interpretation.",
            'psychology': "Participant behavior patterns align with this assessment.",
            'execution': "Risk management protocols should be applied accordingly.",
            'risk': "Continuous monitoring of these risk factors is recommended."
        }
        
        extension = extensions.get(reasoning_type, "Further analysis confirms this assessment.")
        
        # Remove final period and add extension
        if text.endswith('.'):
            text = text[:-1]
        
        return f"{text} {extension}."
    
    def _trim_text(self, text: str, max_length: int) -> str:
        """Trim text while maintaining meaning."""
        if len(text) <= max_length:
            return text
        
        # Split into sentences and keep as many as possible
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        result = ""
        for sentence in sentences:
            test_result = result + sentence + ". " if result else sentence + ". "
            if len(test_result) <= max_length:
                result = test_result
            else:
                break
        
        return result.strip()
    
    def _get_default_text(self, reasoning_type: str) -> str:
        """Get default text when enhancement fails."""
        defaults = {
            'pattern': "Current price action shows standard formation characteristics requiring further confirmation.",
            'context': "Market conditions present balanced technical indicators with neutral directional bias.",
            'psychology': "Participant behavior demonstrates normal risk appetite without extreme sentiment indicators.",
            'execution': "Current setup suggests cautious approach with standard risk management protocols.",
            'risk': "Risk environment remains manageable with normal volatility characteristics."
        }
        
        return defaults.get(reasoning_type, "Technical analysis presents standard market conditions.")
    
    def validate_professional_language(self, text: str) -> Dict[str, Any]:
        """
        Validate text for professional language standards.
        
        Args:
            text: Text to validate
            
        Returns:
            Validation results dictionary
        """
        validation = {
            'is_professional': True,
            'issues': [],
            'score': 100,
            'suggestions': []
        }
        
        # Check for unprofessional terms
        unprofessional_terms = ['gonna', 'wanna', 'gotta', 'yeah', 'nah', 'ok', 'okay']
        for term in unprofessional_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                validation['is_professional'] = False
                validation['issues'].append(f"Contains unprofessional term: {term}")
                validation['score'] -= 20
        
        # Check sentence structure
        if not text.endswith('.'):
            validation['issues'].append("Missing proper sentence ending")
            validation['score'] -= 10
        
        # Check length
        if len(text) < 30:
            validation['issues'].append("Text too short for professional reasoning")
            validation['score'] -= 15
        
        return validation
