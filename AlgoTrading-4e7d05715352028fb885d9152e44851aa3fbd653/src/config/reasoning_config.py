#!/usr/bin/env python3
"""
Reasoning System Configuration
=============================

Configuration file specifically for the reasoning generation system.
Contains all parameters for reasoning engines, text generation, and quality validation.
"""

# === REASONING SYSTEM CONFIGURATION ===
REASONING_CONFIG = {
    # Historical Context Configuration
    'context_window_size': 100,  # Number of previous candles for context analysis
    'context_cache_size': 1000,  # Maximum number of cached context analyses
    
    # Pattern Recognition Engine Configuration
    'pattern_recognition': {
        'support_resistance_threshold': 1.5,  # % distance threshold for S/R proximity
        'pattern_confidence_threshold': 0.7,   # Minimum confidence for pattern recognition
        'use_historical_context': True,        # Include historical pattern analysis
        'confluence_weight': 0.3,              # Weight for confluence scoring
    },
    
    # Context Analysis Engine Configuration
    'context_analysis': {
        'trend_strength_thresholds': {
            'strong': 0.6, 'moderate': 0.3, 'weak': 0.0
        },
        'momentum_thresholds': {
            'overbought': 70, 'oversold': 30, 'neutral_high': 55, 'neutral_low': 45
        },
        'volatility_thresholds': {
            'high': 3.0, 'normal': 1.5, 'low': 0.8
        }
    },
    
    # Psychology Assessment Engine Configuration
    'psychology_assessment': {
        'fear_greed_thresholds': {
            'extreme_fear': 20, 'fear': 35, 'neutral': 65, 'greed': 80, 'extreme_greed': 95
        },
        'momentum_psychology_thresholds': {
            'panic': 15, 'fear': 30, 'neutral_low': 45, 'neutral_high': 55, 'euphoria': 85
        },
        'sentiment_indicators': {
            'rsi': {'extreme_oversold': 20, 'oversold': 30, 'overbought': 70, 'extreme_overbought': 80},
            'williams_r': {'extreme_oversold': -90, 'oversold': -80, 'overbought': -20, 'extreme_overbought': -10},
            'cci': {'extreme_oversold': -200, 'oversold': -100, 'overbought': 100, 'extreme_overbought': 200}
        }
    },
    
    # Execution Decision Engine Configuration
    'execution_decision': {
        'confluence_threshold': 0.7,           # Minimum confluence for strong signals
        'risk_reward_minimum': 1.5,            # Minimum acceptable risk-reward ratio
        'confidence_threshold': 60,            # Minimum confidence for position recommendations
        'position_sizing': {
            'aggressive': 1.0, 'normal': 0.7, 'cautious': 0.4, 'minimal': 0.2
        }
    },
    
    # Risk Assessment Engine Configuration
    'risk_assessment': {
        'volatility_risk_thresholds': {
            'low': 1.0, 'normal': 2.5, 'high': 4.0, 'extreme': 6.0
        },
        'drawdown_thresholds': {
            'low': 2.0, 'moderate': 5.0, 'high': 10.0, 'severe': 20.0
        },
        'scenario_probabilities': {
            'high_confidence': 0.8, 'moderate_confidence': 0.6, 'low_confidence': 0.4
        }
    }
}

# === TEXT GENERATION CONFIGURATION ===
TEXT_GENERATION_CONFIG = {
    # Text Length Configuration
    'min_reasoning_length': 50,               # Minimum characters per reasoning text
    'max_reasoning_length': 300,              # Maximum characters per reasoning text
    'target_reasoning_length': 150,           # Target length for optimal reasoning
    
    # Professional Language Configuration
    'use_professional_enhancement': True,     # Apply professional language enhancement
    'avoid_absolute_prices': True,            # Ensure no absolute price references
    'use_relative_language': True,            # Use relative price descriptions
    'professional_terminology': True,         # Use professional trading terms
    
    # Text Structure Configuration
    'ensure_sentence_structure': True,        # Ensure proper sentence structure
    'capitalize_sentences': True,             # Capitalize first letter of sentences
    'proper_punctuation': True,               # Ensure proper punctuation
    'logical_flow': True,                     # Ensure logical flow between ideas
    
    # Column-Specific Configuration
    'column_requirements': {
        'pattern_recognition_text': {
            'min_length': 50, 'max_length': 250,
            'required_terms': ['pattern', 'formation', 'candle'],
            'focus': 'pattern identification and context'
        },
        'context_analysis_text': {
            'min_length': 60, 'max_length': 300,
            'required_terms': ['market', 'trend', 'technical'],
            'focus': 'market structure and technical confluence'
        },
        'psychology_assessment_text': {
            'min_length': 60, 'max_length': 280,
            'required_terms': ['participant', 'sentiment', 'behavior'],
            'focus': 'market psychology and participant behavior'
        },
        'execution_decision_text': {
            'min_length': 70, 'max_length': 320,
            'required_terms': ['position', 'risk', 'execution'],
            'focus': 'trading decisions and risk management'
        },
        'risk_assessment_text': {
            'min_length': 80, 'max_length': 350,
            'required_terms': ['risk', 'scenario', 'potential'],
            'focus': 'risk analysis and scenario planning'
        },
        'alternative_scenarios_text': {
            'min_length': 60, 'max_length': 300,
            'required_terms': ['scenario', 'alternative', 'consider'],
            'focus': 'alternative interpretations and scenarios'
        }
    }
}

# === QUALITY VALIDATION CONFIGURATION ===
QUALITY_VALIDATION_CONFIG = {
    # Overall Quality Standards
    'min_quality_score': 70,                 # Minimum acceptable quality score
    'target_quality_score': 85,              # Target quality score
    'excellent_quality_score': 95,           # Excellent quality threshold
    
    # Validation Checks
    'check_price_references': True,          # Check for absolute price references
    'check_logical_consistency': True,       # Check cross-column consistency
    'check_professional_language': True,     # Check professional language usage
    'check_text_structure': True,            # Check text structure and formatting
    'check_required_patterns': True,         # Check for required terms/patterns
    
    # Price Reference Validation
    'forbidden_price_patterns': [
        r'\b\d{4,5}\.\d+\b',                 # Prices like 23450.50
        r'\b\d{4,5}\b',                      # Prices like 23450
        r'\$\d+',                            # Dollar amounts
        r'₹\d+',                             # Rupee amounts
        r'\b\d+\s*points?\b',                # Point references
    ],
    
    # Professional Language Validation
    'unprofessional_terms': [
        'gonna', 'wanna', 'gotta', 'yeah', 'nah', 'ok', 'okay', 'stuff', 'things'
    ],
    
    # Consistency Validation
    'sentiment_consistency_weight': 0.3,     # Weight for sentiment consistency
    'risk_reward_alignment_weight': 0.4,     # Weight for risk-reward alignment
    'pattern_context_alignment_weight': 0.3, # Weight for pattern-context alignment
    
    # Quality Scoring
    'quality_weights': {
        'professional_language': 0.25,       # Weight for professional language
        'text_structure': 0.20,              # Weight for text structure
        'required_patterns': 0.15,           # Weight for required patterns
        'price_references': 0.20,            # Weight for price reference compliance
        'logical_consistency': 0.20          # Weight for logical consistency
    }
}

# === PROCESSING CONFIGURATION ===
PROCESSING_CONFIG = {
    # Batch Processing
    'batch_size': 1000,                      # Rows to process in each batch
    'progress_reporting_interval': 1000,     # Report progress every N rows
    'memory_optimization': True,             # Enable memory optimization
    
    # File Processing
    'input_file_pattern': 'features_*.csv',  # Pattern for input files
    'output_file_prefix': 'reasoning_',      # Prefix for output files
    'backup_original_files': False,          # Backup original files before processing
    
    # Error Handling
    'continue_on_error': True,               # Continue processing if individual rows fail
    'max_errors_per_file': 100,             # Maximum errors before stopping file processing
    'log_detailed_errors': True,            # Log detailed error information
    
    # Performance
    'parallel_processing': False,            # Enable parallel processing (experimental)
    'cache_context_analysis': True,         # Cache context analysis for performance
    'optimize_memory_usage': True,          # Optimize memory usage for large files
    
    # Output Configuration
    'save_quality_reports': True,           # Save detailed quality reports
    'generate_summary_statistics': True,    # Generate summary statistics
    'include_debug_columns': False,         # Include debug columns in output
}

# === LOGGING CONFIGURATION ===
LOGGING_CONFIG = {
    'level': 'INFO',                        # Logging level (DEBUG, INFO, WARNING, ERROR)
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,                    # Log to file
    'log_file': 'reasoning_system.log',     # Log file name
    'max_log_size': 10 * 1024 * 1024,      # Maximum log file size (10MB)
    'backup_count': 5,                      # Number of backup log files
    'log_reasoning_generation': False,      # Log individual reasoning generation (verbose)
    'log_quality_validation': False,       # Log quality validation details (verbose)
}

def get_reasoning_config():
    """
    Get the complete reasoning system configuration.
    
    Returns:
        dict: Complete configuration with all sections
    """
    return {
        'reasoning': REASONING_CONFIG,
        'text_generation': TEXT_GENERATION_CONFIG,
        'quality_validation': QUALITY_VALIDATION_CONFIG,
        'processing': PROCESSING_CONFIG,
        'logging': LOGGING_CONFIG,
    }

def print_reasoning_config():
    """Print the reasoning system configuration in a readable format."""
    config = get_reasoning_config()
    
    print("=" * 80)
    print("REASONING SYSTEM CONFIGURATION")
    print("=" * 80)
    
    for section_name, section_config in config.items():
        print(f"\n[{section_name.upper()}]")
        
        if isinstance(section_config, dict):
            for key, value in section_config.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {section_config}")
    
    print("\n" + "=" * 80)

def validate_reasoning_config():
    """Validate the reasoning system configuration."""
    config = get_reasoning_config()
    issues = []
    
    # Validate context window size
    if config['reasoning']['context_window_size'] < 10:
        issues.append("Context window size too small (minimum 10)")
    
    # Validate quality thresholds
    if config['quality_validation']['min_quality_score'] > 100:
        issues.append("Minimum quality score cannot exceed 100")
    
    # Validate text length constraints
    text_config = config['text_generation']
    if text_config['min_reasoning_length'] >= text_config['max_reasoning_length']:
        issues.append("Minimum reasoning length must be less than maximum")
    
    # Validate processing configuration
    if config['processing']['batch_size'] < 1:
        issues.append("Batch size must be at least 1")
    
    if issues:
        print("Configuration validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("Configuration validation passed")
    return True

if __name__ == "__main__":
    print_reasoning_config()
    print()
    validate_reasoning_config()
