#!/usr/bin/env python3
"""
Quality Validator
================

Validates the quality and consistency of generated reasoning text across
all reasoning components and provides quality metrics.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class QualityValidator:
    """
    Validates reasoning quality across multiple dimensions including
    consistency, coherence, professional language, and logical flow.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the quality validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Load validation rules and patterns
        self.forbidden_patterns = self._load_forbidden_patterns()
        self.required_patterns = self._load_required_patterns()
        self.consistency_rules = self._load_consistency_rules()
        
        # Configuration
        self.config.setdefault('min_quality_score', 70)
        self.config.setdefault('check_price_references', True)
        self.config.setdefault('check_logical_consistency', True)
        self.config.setdefault('check_professional_language', True)
        
        logger.info("QualityValidator initialized with validation rules")
    
    def validate_reasoning_row(self, reasoning_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single row of reasoning data.
        
        Args:
            reasoning_row: Dictionary containing all reasoning columns for one row
            
        Returns:
            Validation results for the row
        """
        validation = {
            'overall_score': 0,
            'individual_scores': {},
            'issues': [],
            'warnings': [],
            'passed_checks': [],
            'failed_checks': []
        }
        
        try:
            # Validate each reasoning column
            reasoning_columns = [
                'pattern_recognition_text',
                'context_analysis_text', 
                'psychology_assessment_text',
                'execution_decision_text',
                'risk_assessment_text',
                'alternative_scenarios_text'
            ]
            
            column_scores = []
            
            for column in reasoning_columns:
                if column in reasoning_row:
                    text = reasoning_row[column]
                    score = self._validate_single_text(text, column)
                    validation['individual_scores'][column] = score
                    column_scores.append(score)
                else:
                    validation['issues'].append(f"Missing reasoning column: {column}")
                    column_scores.append(0)
            
            # Validate confidence score
            if 'confidence_score' in reasoning_row:
                conf_score = reasoning_row['confidence_score']
                conf_validation = self._validate_confidence_score(conf_score)
                validation['individual_scores']['confidence_score'] = conf_validation['score']
                column_scores.append(conf_validation['score'])
                
                if conf_validation['issues']:
                    validation['issues'].extend(conf_validation['issues'])
            else:
                validation['issues'].append("Missing confidence score")
                column_scores.append(0)
            
            # Cross-column consistency validation
            consistency_score = self._validate_cross_column_consistency(reasoning_row)
            validation['individual_scores']['consistency'] = consistency_score
            column_scores.append(consistency_score)
            
            # Calculate overall score
            validation['overall_score'] = np.mean(column_scores) if column_scores else 0
            
            # Determine pass/fail status
            min_score = self.config['min_quality_score']
            if validation['overall_score'] >= min_score:
                validation['passed_checks'].append('overall_quality')
            else:
                validation['failed_checks'].append('overall_quality')
            
        except Exception as e:
            logger.error(f"Error validating reasoning row: {str(e)}")
            validation['issues'].append(f"Validation error: {str(e)}")
            validation['overall_score'] = 0
        
        return validation
    
    def validate_reasoning_dataframe(self, reasoning_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate entire DataFrame of reasoning data.
        
        Args:
            reasoning_df: DataFrame containing reasoning columns
            
        Returns:
            Comprehensive validation report
        """
        report = {
            'total_rows': len(reasoning_df),
            'overall_score': 0,
            'column_scores': {},
            'quality_distribution': {},
            'common_issues': {},
            'recommendations': []
        }
        
        try:
            # Validate each row
            row_scores = []
            all_issues = []
            column_score_lists = {}
            
            for idx, row in reasoning_df.iterrows():
                if idx % 1000 == 0:
                    logger.info(f"Validating row {idx}/{len(reasoning_df)}")
                
                row_validation = self.validate_reasoning_row(row.to_dict())
                row_scores.append(row_validation['overall_score'])
                all_issues.extend(row_validation['issues'])
                
                # Collect column scores
                for col, score in row_validation['individual_scores'].items():
                    if col not in column_score_lists:
                        column_score_lists[col] = []
                    column_score_lists[col].append(score)
            
            # Calculate summary statistics
            report['overall_score'] = np.mean(row_scores) if row_scores else 0
            
            # Column-wise scores
            for col, scores in column_score_lists.items():
                report['column_scores'][col] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
            
            # Quality distribution
            score_ranges = [(0, 30), (30, 50), (50, 70), (70, 85), (85, 100)]
            for low, high in score_ranges:
                count = sum(1 for score in row_scores if low <= score < high)
                report['quality_distribution'][f'{low}-{high}'] = count
            
            # Common issues analysis
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            # Top 10 most common issues
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            report['common_issues'] = dict(sorted_issues[:10])
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(report)
            
        except Exception as e:
            logger.error(f"Error validating reasoning DataFrame: {str(e)}")
            report['overall_score'] = 0
            report['recommendations'] = [f"Validation failed: {str(e)}"]
        
        return report
    
    def _validate_single_text(self, text: str, column_type: str) -> float:
        """
        Validate a single piece of reasoning text.
        
        Args:
            text: Text to validate
            column_type: Type of reasoning column
            
        Returns:
            Quality score (0-100)
        """
        if not text or not text.strip():
            return 0
        
        score = 100
        issues = []
        
        # Check for forbidden patterns (absolute price references)
        if self.config['check_price_references']:
            price_issues = self._check_price_references(text)
            if price_issues:
                score -= len(price_issues) * 20
                issues.extend(price_issues)
        
        # Check professional language
        if self.config['check_professional_language']:
            lang_score = self._check_professional_language(text)
            score = min(score, lang_score)
        
        # Check text length and structure
        structure_score = self._check_text_structure(text, column_type)
        score = min(score, structure_score)
        
        # Check for required patterns
        required_score = self._check_required_patterns(text, column_type)
        score = min(score, required_score)
        
        return max(0, score)
    
    def _validate_confidence_score(self, confidence_score: Any) -> Dict[str, Any]:
        """
        Validate confidence score value.
        
        Args:
            confidence_score: Confidence score to validate
            
        Returns:
            Validation results for confidence score
        """
        validation = {
            'score': 100,
            'issues': []
        }
        
        try:
            # Check if it's a valid number
            score = float(confidence_score)
            
            # Check range
            if not (0 <= score <= 100):
                validation['issues'].append(f"Confidence score {score} outside valid range 0-100")
                validation['score'] = 0
            
            # Check if it's an integer (as expected)
            if score != int(score):
                validation['issues'].append(f"Confidence score {score} should be integer")
                validation['score'] = 80
            
        except (ValueError, TypeError):
            validation['issues'].append(f"Invalid confidence score type: {type(confidence_score)}")
            validation['score'] = 0
        
        return validation
    
    def _validate_cross_column_consistency(self, reasoning_row: Dict[str, Any]) -> float:
        """
        Validate consistency across reasoning columns.
        
        Args:
            reasoning_row: Dictionary containing all reasoning columns
            
        Returns:
            Consistency score (0-100)
        """
        if not self.config['check_logical_consistency']:
            return 100
        
        score = 100
        
        # Check for contradictory sentiment across columns
        sentiment_words = {
            'bullish': ['bullish', 'upward', 'positive', 'buying', 'support', 'advance'],
            'bearish': ['bearish', 'downward', 'negative', 'selling', 'resistance', 'decline'],
            'neutral': ['neutral', 'balanced', 'mixed', 'uncertain', 'consolidation']
        }
        
        column_sentiments = {}
        
        for col_name, text in reasoning_row.items():
            if isinstance(text, str) and col_name.endswith('_text'):
                text_lower = text.lower()
                
                bullish_count = sum(1 for word in sentiment_words['bullish'] if word in text_lower)
                bearish_count = sum(1 for word in sentiment_words['bearish'] if word in text_lower)
                neutral_count = sum(1 for word in sentiment_words['neutral'] if word in text_lower)
                
                if bullish_count > bearish_count and bullish_count > neutral_count:
                    column_sentiments[col_name] = 'bullish'
                elif bearish_count > bullish_count and bearish_count > neutral_count:
                    column_sentiments[col_name] = 'bearish'
                else:
                    column_sentiments[col_name] = 'neutral'
        
        # Check for major contradictions
        sentiments = list(column_sentiments.values())
        if len(set(sentiments)) > 2:  # More than 2 different sentiments
            score -= 30
        
        return max(0, score)
    
    def _check_price_references(self, text: str) -> List[str]:
        """Check for absolute price references that should be avoided."""
        issues = []
        
        # Pattern for absolute price references
        price_patterns = [
            r'\b\d{4,5}\.\d+\b',  # Prices like 23450.50
            r'\b\d{4,5}\b',       # Prices like 23450
            r'\$\d+',             # Dollar amounts
            r'₹\d+',              # Rupee amounts
            r'\b\d+\s*points?\b', # Point references
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            if matches:
                issues.append(f"Contains absolute price reference: {matches[0]}")
        
        return issues
    
    def _check_professional_language(self, text: str) -> float:
        """Check for professional language usage."""
        score = 100
        
        # Unprofessional terms
        unprofessional = ['gonna', 'wanna', 'gotta', 'yeah', 'nah', 'ok', 'okay', 'stuff', 'things']
        
        for term in unprofessional:
            if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                score -= 20
        
        # Check for proper sentence structure
        if not text.strip().endswith('.'):
            score -= 10
        
        # Check for proper capitalization
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence[0].isupper():
                score -= 5
                break
        
        return max(0, score)
    
    def _check_text_structure(self, text: str, column_type: str) -> float:
        """Check text structure and length."""
        score = 100
        
        # Length checks
        min_lengths = {
            'pattern_recognition_text': 50,
            'context_analysis_text': 60,
            'psychology_assessment_text': 60,
            'execution_decision_text': 70,
            'risk_assessment_text': 80,
            'alternative_scenarios_text': 60
        }
        
        max_length = 400
        min_length = min_lengths.get(column_type, 50)
        
        if len(text) < min_length:
            score -= 30
        elif len(text) > max_length:
            score -= 20
        
        # Check for proper sentence structure
        if text.count('.') < 1:
            score -= 20
        
        return max(0, score)
    
    def _check_required_patterns(self, text: str, column_type: str) -> float:
        """Check for required patterns based on column type."""
        score = 100
        
        required_terms = {
            'pattern_recognition_text': ['pattern', 'formation', 'candle'],
            'context_analysis_text': ['market', 'trend', 'technical'],
            'psychology_assessment_text': ['participant', 'sentiment', 'behavior'],
            'execution_decision_text': ['position', 'risk', 'execution'],
            'risk_assessment_text': ['risk', 'scenario', 'potential'],
            'alternative_scenarios_text': ['scenario', 'alternative', 'consider']
        }
        
        if column_type in required_terms:
            required = required_terms[column_type]
            text_lower = text.lower()
            
            missing_terms = []
            for term in required:
                if term not in text_lower:
                    missing_terms.append(term)
            
            if missing_terms:
                score -= len(missing_terms) * 15
        
        return max(0, score)
    
    def _load_forbidden_patterns(self) -> List[str]:
        """Load patterns that should not appear in reasoning text."""
        return [
            r'\b\d{4,5}\.\d+\b',  # Absolute prices
            r'\b\d{4,5}\b',       # Absolute price levels
            r'exactly \d+',       # Exact numbers
            r'precisely \d+',     # Precise numbers
        ]
    
    def _load_required_patterns(self) -> Dict[str, List[str]]:
        """Load patterns that should appear in specific reasoning types."""
        return {
            'professional_terms': ['analysis', 'indicates', 'suggests', 'demonstrates'],
            'relative_language': ['current', 'recent', 'key level', 'support zone'],
            'uncertainty_expressions': ['potential', 'possible', 'likely', 'suggests']
        }
    
    def _load_consistency_rules(self) -> List[Dict[str, Any]]:
        """Load rules for cross-column consistency checking."""
        return [
            {
                'name': 'sentiment_consistency',
                'description': 'Sentiment should be consistent across columns',
                'weight': 0.3
            },
            {
                'name': 'risk_reward_alignment',
                'description': 'Risk assessment should align with execution decision',
                'weight': 0.4
            },
            {
                'name': 'pattern_context_alignment',
                'description': 'Pattern recognition should align with context analysis',
                'weight': 0.3
            }
        ]
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        overall_score = report['overall_score']
        
        if overall_score < 50:
            recommendations.append("Overall quality is poor. Consider reviewing reasoning generation logic.")
        elif overall_score < 70:
            recommendations.append("Quality is below target. Focus on improving consistency and professional language.")
        
        # Column-specific recommendations
        for col, scores in report['column_scores'].items():
            if scores['mean'] < 60:
                recommendations.append(f"Improve {col} quality - current average: {scores['mean']:.1f}")
        
        # Common issues recommendations
        common_issues = report['common_issues']
        if common_issues:
            top_issue = max(common_issues.items(), key=lambda x: x[1])
            recommendations.append(f"Address most common issue: {top_issue[0]} (occurs {top_issue[1]} times)")
        
        return recommendations
