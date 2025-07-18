#!/usr/bin/env python3
"""
Template Variations for Enhanced Reasoning System
================================================

Provides 50+ different pattern recognition templates with variations
to achieve >60% unique content vs current 4.4%.

Key Features:
- Multiple variations for same concepts (5-10 ways to express each idea)
- Contextual phrase selection based on market conditions
- Dynamic language intensity based on market strength
- Natural language diversity for pattern recognition
"""

import random
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class TemplateVariations:
    """
    Template variation system for diverse natural language generation.
    """
    
    def __init__(self):
        """Initialize template variations."""
        self.pattern_templates = self._load_pattern_templates()
        self.context_templates = self._load_context_templates()
        self.psychology_templates = self._load_psychology_templates()
        self.execution_templates = self._load_execution_templates()
        self.risk_templates = self._load_risk_templates()
        
        # Language intensity modifiers
        self.intensity_modifiers = self._load_intensity_modifiers()
        
        # Contextual connectors
        self.connectors = self._load_contextual_connectors()
        
        logger.info("TemplateVariations initialized with 50+ template variations")
    
    def get_pattern_template(self, pattern_type: str, market_strength: float = 0.5) -> Dict[str, Any]:
        """
        Get pattern template with variations based on market strength.
        
        Args:
            pattern_type: Type of pattern (bullish, bearish, neutral, etc.)
            market_strength: Market strength (0-1) for intensity selection
            
        Returns:
            Template dictionary with variations
        """
        templates = self.pattern_templates.get(pattern_type, self.pattern_templates['neutral'])
        base_template = random.choice(templates)
        
        # Apply intensity modifiers based on market strength
        intensity = self._determine_intensity(market_strength)
        modified_template = self._apply_intensity_modifiers(base_template, intensity)
        
        return modified_template
    
    def get_context_template(self, context_type: str, confidence_level: float = 0.5) -> Dict[str, Any]:
        """Get context template with confidence-based variations."""
        templates = self.context_templates.get(context_type, self.context_templates['balanced'])
        base_template = random.choice(templates)
        
        # Apply confidence modifiers
        confidence_category = self._determine_confidence_category(confidence_level)
        modified_template = self._apply_confidence_modifiers(base_template, confidence_category)
        
        return modified_template
    
    def _load_pattern_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load pattern recognition templates with variations."""
        return {
            'bullish_momentum': [
                {
                    'base': "Price action demonstrates {intensity} bullish momentum characteristics",
                    'variations': [
                        "Current candle formation reveals {intensity} upward momentum development",
                        "Technical analysis shows {intensity} bullish momentum acceleration",
                        "Market structure exhibits {intensity} positive momentum characteristics",
                        "Price behavior indicates {intensity} bullish momentum building",
                        "Momentum indicators confirm {intensity} upward pressure development"
                    ],
                    'connectors': ["with", "alongside", "supported by", "confirmed by", "reinforced by"]
                },
                {
                    'base': "Bullish pattern formation shows {intensity} continuation potential",
                    'variations': [
                        "Upward pattern development suggests {intensity} momentum persistence",
                        "Positive formation characteristics indicate {intensity} trend continuation",
                        "Bullish structure evolution demonstrates {intensity} follow-through potential",
                        "Constructive pattern behavior reveals {intensity} upside momentum",
                        "Favorable formation dynamics suggest {intensity} bullish persistence"
                    ],
                    'connectors': ["indicating", "suggesting", "implying", "demonstrating", "revealing"]
                }
            ],
            'bearish_momentum': [
                {
                    'base': "Price action demonstrates {intensity} bearish momentum characteristics",
                    'variations': [
                        "Current candle formation reveals {intensity} downward momentum development",
                        "Technical analysis shows {intensity} bearish momentum acceleration",
                        "Market structure exhibits {intensity} negative momentum characteristics",
                        "Price behavior indicates {intensity} bearish momentum building",
                        "Momentum indicators confirm {intensity} downward pressure development"
                    ],
                    'connectors': ["with", "alongside", "supported by", "confirmed by", "reinforced by"]
                },
                {
                    'base': "Bearish pattern formation shows {intensity} continuation potential",
                    'variations': [
                        "Downward pattern development suggests {intensity} momentum persistence",
                        "Negative formation characteristics indicate {intensity} trend continuation",
                        "Bearish structure evolution demonstrates {intensity} follow-through potential",
                        "Destructive pattern behavior reveals {intensity} downside momentum",
                        "Unfavorable formation dynamics suggest {intensity} bearish persistence"
                    ],
                    'connectors': ["indicating", "suggesting", "implying", "demonstrating", "revealing"]
                }
            ],
            'consolidation': [
                {
                    'base': "Price action shows {intensity} consolidation characteristics",
                    'variations': [
                        "Market behavior exhibits {intensity} range-bound tendencies",
                        "Technical structure demonstrates {intensity} sideways movement patterns",
                        "Price dynamics reveal {intensity} consolidation phase development",
                        "Market action indicates {intensity} horizontal trading behavior",
                        "Formation characteristics suggest {intensity} range-bound conditions"
                    ],
                    'connectors': ["within", "during", "throughout", "across", "spanning"]
                },
                {
                    'base': "Consolidation pattern indicates {intensity} directional uncertainty",
                    'variations': [
                        "Range-bound behavior suggests {intensity} market indecision",
                        "Sideways movement reflects {intensity} directional equilibrium",
                        "Horizontal pattern development shows {intensity} balanced conditions",
                        "Consolidation dynamics indicate {intensity} participant uncertainty",
                        "Range formation reveals {intensity} directional pause"
                    ],
                    'connectors': ["pending", "awaiting", "preceding", "before", "until"]
                }
            ],
            'reversal_potential': [
                {
                    'base': "Pattern formation suggests {intensity} reversal potential",
                    'variations': [
                        "Technical structure indicates {intensity} directional change possibility",
                        "Formation characteristics reveal {intensity} momentum shift potential",
                        "Pattern development suggests {intensity} trend reversal setup",
                        "Market structure shows {intensity} directional transition signals",
                        "Formation dynamics indicate {intensity} reversal pattern emergence"
                    ],
                    'connectors': ["following", "after", "subsequent to", "in response to", "triggered by"]
                },
                {
                    'base': "Reversal signals show {intensity} momentum shift characteristics",
                    'variations': [
                        "Directional change indicators reveal {intensity} momentum transition",
                        "Reversal formation demonstrates {intensity} trend shift potential",
                        "Pattern reversal suggests {intensity} directional momentum change",
                        "Formation reversal indicates {intensity} trend transition development",
                        "Momentum reversal signals show {intensity} directional shift"
                    ],
                    'connectors': ["as", "while", "during", "throughout", "amid"]
                }
            ],
            'breakout_potential': [
                {
                    'base': "Formation suggests {intensity} breakout potential",
                    'variations': [
                        "Pattern development indicates {intensity} range expansion possibility",
                        "Technical structure shows {intensity} directional breakout setup",
                        "Formation characteristics reveal {intensity} volatility expansion potential",
                        "Pattern evolution suggests {intensity} range resolution development",
                        "Structure dynamics indicate {intensity} breakout preparation"
                    ],
                    'connectors': ["approaching", "nearing", "building toward", "developing into", "evolving toward"]
                },
                {
                    'base': "Breakout setup shows {intensity} directional resolution potential",
                    'variations': [
                        "Range resolution indicates {intensity} volatility expansion setup",
                        "Directional setup suggests {intensity} momentum acceleration potential",
                        "Breakout formation reveals {intensity} trend establishment possibility",
                        "Resolution pattern shows {intensity} directional commitment development",
                        "Expansion setup indicates {intensity} volatility breakout potential"
                    ],
                    'connectors': ["leading to", "resulting in", "culminating in", "developing into", "transitioning to"]
                }
            ],
            'neutral': [
                {
                    'base': "Price action shows {intensity} balanced characteristics",
                    'variations': [
                        "Market behavior exhibits {intensity} equilibrium conditions",
                        "Technical structure demonstrates {intensity} neutral positioning",
                        "Price dynamics reveal {intensity} balanced market conditions",
                        "Formation characteristics indicate {intensity} market equilibrium",
                        "Pattern development suggests {intensity} directional balance"
                    ],
                    'connectors': ["maintaining", "preserving", "sustaining", "continuing", "extending"]
                }
            ]
        }
    
    def _load_context_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load context analysis templates with variations."""
        return {
            'trending_market': [
                {
                    'base': "Market environment demonstrates {confidence} trending characteristics",
                    'variations': [
                        "Current market regime exhibits {confidence} directional bias",
                        "Market structure shows {confidence} trending behavior patterns",
                        "Trading environment reveals {confidence} directional momentum",
                        "Market conditions indicate {confidence} trend-following dynamics",
                        "Current regime demonstrates {confidence} directional persistence"
                    ]
                },
                {
                    'base': "Trending conditions support {confidence} directional strategies",
                    'variations': [
                        "Directional environment favors {confidence} momentum approaches",
                        "Trending regime supports {confidence} trend-following tactics",
                        "Market bias enables {confidence} directional positioning",
                        "Trending dynamics facilitate {confidence} momentum strategies",
                        "Directional conditions encourage {confidence} trend-based approaches"
                    ]
                }
            ],
            'ranging_market': [
                {
                    'base': "Market environment shows {confidence} range-bound characteristics",
                    'variations': [
                        "Current market regime exhibits {confidence} sideways behavior",
                        "Market structure demonstrates {confidence} horizontal dynamics",
                        "Trading environment reveals {confidence} range-bound conditions",
                        "Market conditions indicate {confidence} consolidation patterns",
                        "Current regime shows {confidence} mean-reversion tendencies"
                    ]
                }
            ],
            'volatile_market': [
                {
                    'base': "Market environment exhibits {confidence} elevated volatility",
                    'variations': [
                        "Current market regime shows {confidence} increased uncertainty",
                        "Market structure demonstrates {confidence} heightened volatility",
                        "Trading environment reveals {confidence} expanded price ranges",
                        "Market conditions indicate {confidence} elevated risk levels",
                        "Current regime exhibits {confidence} amplified price movement"
                    ]
                }
            ],
            'balanced': [
                {
                    'base': "Market environment shows {confidence} balanced conditions",
                    'variations': [
                        "Current market regime exhibits {confidence} equilibrium characteristics",
                        "Market structure demonstrates {confidence} neutral positioning",
                        "Trading environment reveals {confidence} balanced dynamics",
                        "Market conditions indicate {confidence} stable characteristics",
                        "Current regime shows {confidence} measured behavior"
                    ]
                }
            ]
        }

    def _load_psychology_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load psychology assessment templates with variations."""
        return {
            'bullish_sentiment': [
                {
                    'base': "Market participants demonstrate {intensity} bullish sentiment",
                    'variations': [
                        "Trader behavior exhibits {intensity} positive market sentiment",
                        "Participant psychology shows {intensity} optimistic characteristics",
                        "Market sentiment reflects {intensity} bullish participant behavior",
                        "Trader psychology indicates {intensity} positive market outlook",
                        "Participant sentiment demonstrates {intensity} constructive behavior"
                    ]
                }
            ],
            'bearish_sentiment': [
                {
                    'base': "Market participants demonstrate {intensity} bearish sentiment",
                    'variations': [
                        "Trader behavior exhibits {intensity} negative market sentiment",
                        "Participant psychology shows {intensity} pessimistic characteristics",
                        "Market sentiment reflects {intensity} bearish participant behavior",
                        "Trader psychology indicates {intensity} negative market outlook",
                        "Participant sentiment demonstrates {intensity} defensive behavior"
                    ]
                }
            ],
            'neutral_sentiment': [
                {
                    'base': "Market participants show {intensity} balanced sentiment",
                    'variations': [
                        "Trader behavior exhibits {intensity} neutral market sentiment",
                        "Participant psychology demonstrates {intensity} equilibrium characteristics",
                        "Market sentiment reflects {intensity} balanced participant behavior",
                        "Trader psychology indicates {intensity} measured market outlook",
                        "Participant sentiment shows {intensity} cautious behavior"
                    ]
                }
            ]
        }

    def _load_execution_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load execution decision templates with variations."""
        return {
            'aggressive_positioning': [
                {
                    'base': "Current analysis supports {intensity} aggressive positioning",
                    'variations': [
                        "Market conditions favor {intensity} decisive position initiation",
                        "Technical setup enables {intensity} confident position establishment",
                        "Analysis supports {intensity} assertive market positioning",
                        "Conditions warrant {intensity} proactive position building",
                        "Setup justifies {intensity} aggressive market participation"
                    ]
                }
            ],
            'cautious_positioning': [
                {
                    'base': "Current analysis suggests {intensity} cautious positioning",
                    'variations': [
                        "Market conditions recommend {intensity} selective position initiation",
                        "Technical setup indicates {intensity} measured position establishment",
                        "Analysis supports {intensity} conservative market positioning",
                        "Conditions warrant {intensity} defensive position building",
                        "Setup justifies {intensity} prudent market participation"
                    ]
                }
            ],
            'wait_and_see': [
                {
                    'base': "Current analysis recommends {intensity} patient approach",
                    'variations': [
                        "Market conditions suggest {intensity} watchful waiting strategy",
                        "Technical setup indicates {intensity} observational positioning",
                        "Analysis supports {intensity} cautious market monitoring",
                        "Conditions warrant {intensity} selective opportunity assessment",
                        "Setup justifies {intensity} patient market evaluation"
                    ]
                }
            ]
        }

    def _load_risk_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load risk assessment templates with variations."""
        return {
            'low_risk': [
                {
                    'base': "Risk assessment indicates {intensity} favorable conditions",
                    'variations': [
                        "Risk analysis shows {intensity} manageable exposure levels",
                        "Risk evaluation reveals {intensity} acceptable risk parameters",
                        "Risk assessment demonstrates {intensity} controlled risk environment",
                        "Risk analysis indicates {intensity} favorable risk-reward dynamics",
                        "Risk evaluation shows {intensity} measured risk characteristics"
                    ]
                }
            ],
            'moderate_risk': [
                {
                    'base': "Risk assessment shows {intensity} balanced risk conditions",
                    'variations': [
                        "Risk analysis indicates {intensity} moderate exposure levels",
                        "Risk evaluation reveals {intensity} standard risk parameters",
                        "Risk assessment demonstrates {intensity} typical risk environment",
                        "Risk analysis shows {intensity} normal risk-reward dynamics",
                        "Risk evaluation indicates {intensity} conventional risk characteristics"
                    ]
                }
            ],
            'high_risk': [
                {
                    'base': "Risk assessment indicates {intensity} elevated risk conditions",
                    'variations': [
                        "Risk analysis shows {intensity} heightened exposure levels",
                        "Risk evaluation reveals {intensity} challenging risk parameters",
                        "Risk assessment demonstrates {intensity} demanding risk environment",
                        "Risk analysis indicates {intensity} unfavorable risk-reward dynamics",
                        "Risk evaluation shows {intensity} elevated risk characteristics"
                    ]
                }
            ]
        }

    def _load_intensity_modifiers(self) -> Dict[str, List[str]]:
        """Load intensity modifiers for dynamic language."""
        return {
            'weak': ['subtle', 'modest', 'limited', 'restrained', 'measured'],
            'moderate': ['moderate', 'balanced', 'steady', 'consistent', 'stable'],
            'strong': ['pronounced', 'significant', 'notable', 'substantial', 'compelling'],
            'very_strong': ['exceptional', 'remarkable', 'outstanding', 'extraordinary', 'powerful']
        }

    def _load_contextual_connectors(self) -> Dict[str, List[str]]:
        """Load contextual connectors for natural flow."""
        return {
            'continuation': ['while', 'as', 'with', 'alongside', 'during'],
            'contrast': ['however', 'nevertheless', 'although', 'despite', 'yet'],
            'causation': ['therefore', 'consequently', 'thus', 'resulting in', 'leading to'],
            'addition': ['furthermore', 'additionally', 'moreover', 'also', 'in addition'],
            'temporal': ['currently', 'presently', 'meanwhile', 'simultaneously', 'concurrently']
        }

    def _determine_intensity(self, market_strength: float) -> str:
        """Determine intensity category based on market strength."""
        if market_strength > 0.8:
            return 'very_strong'
        elif market_strength > 0.6:
            return 'strong'
        elif market_strength > 0.4:
            return 'moderate'
        else:
            return 'weak'

    def _determine_confidence_category(self, confidence_level: float) -> str:
        """Determine confidence category."""
        if confidence_level > 0.8:
            return 'high'
        elif confidence_level > 0.6:
            return 'moderate'
        else:
            return 'low'

    def _apply_intensity_modifiers(self, template: Dict[str, Any], intensity: str) -> Dict[str, Any]:
        """Apply intensity modifiers to template."""
        modified_template = template.copy()

        # Select random intensity modifier
        intensity_word = random.choice(self.intensity_modifiers[intensity])

        # Apply to base template
        if 'base' in modified_template:
            modified_template['base'] = modified_template['base'].format(intensity=intensity_word)

        # Apply to variations
        if 'variations' in modified_template:
            modified_template['variations'] = [
                var.format(intensity=intensity_word) for var in modified_template['variations']
            ]

        return modified_template

    def _apply_confidence_modifiers(self, template: Dict[str, Any], confidence_category: str) -> Dict[str, Any]:
        """Apply confidence modifiers to template."""
        modified_template = template.copy()

        confidence_modifiers = {
            'high': ['clear', 'definitive', 'strong', 'convincing', 'compelling'],
            'moderate': ['reasonable', 'adequate', 'sufficient', 'acceptable', 'moderate'],
            'low': ['limited', 'tentative', 'uncertain', 'questionable', 'weak']
        }

        confidence_word = random.choice(confidence_modifiers[confidence_category])

        # Apply to base template
        if 'base' in modified_template:
            modified_template['base'] = modified_template['base'].format(confidence=confidence_word)

        # Apply to variations
        if 'variations' in modified_template:
            modified_template['variations'] = [
                var.format(confidence=confidence_word) for var in modified_template['variations']
            ]

        return modified_template

    def get_random_variation(self, template: Dict[str, Any]) -> str:
        """Get random variation from template."""
        if 'variations' in template and template['variations']:
            return random.choice(template['variations'])
        elif 'base' in template:
            return template['base']
        else:
            return "Technical analysis indicates current market conditions"

    def get_contextual_connector(self, connector_type: str = 'continuation') -> str:
        """Get contextual connector for natural flow."""
        connectors = self.connectors.get(connector_type, self.connectors['continuation'])
        return random.choice(connectors)
