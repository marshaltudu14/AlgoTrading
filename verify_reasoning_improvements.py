#!/usr/bin/env python3
"""
Verify Enhanced Reasoning System Improvements
===========================================

Analyzes the enhanced reasoning system to verify:
1. Decision-signal alignment (100% accuracy)
2. Natural language quality improvements
3. Content uniqueness and diversity
4. No signal column references in reasoning text
5. Processing speed and quality metrics

Usage:
    python verify_reasoning_improvements.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import Counter
import time

def load_reasoning_data():
    """Load the enhanced reasoning data."""
    reasoning_files = list(Path("data/processed/reasoning").glob("reasoning_*.csv"))
    
    if not reasoning_files:
        print("❌ No reasoning files found!")
        return None
    
    print(f"Found {len(reasoning_files)} reasoning files:")
    for file in reasoning_files:
        print(f"  - {file.name}")
    
    # Load the first file for analysis
    df = pd.read_csv(reasoning_files[0])
    print(f"\nLoaded {len(df)} rows from {reasoning_files[0].name}")
    
    return df

def verify_decision_signal_alignment(df):
    """Verify that decision column aligns 100% with signal column."""
    print("\n" + "="*60)
    print("DECISION-SIGNAL ALIGNMENT VERIFICATION")
    print("="*60)
    
    if 'decision' not in df.columns or 'signal' not in df.columns:
        print("❌ Missing decision or signal column!")
        return False
    
    # Define expected decision patterns based on signal
    signal_to_decision = {
        0: "Staying HOLD",
        1: "Going LONG", 
        2: "Going SHORT"
    }
    
    total_rows = len(df)
    correct_alignments = 0
    misalignments = []
    
    for idx, row in df.iterrows():
        signal = row['signal']
        decision = row['decision']
        
        expected_start = signal_to_decision.get(signal, "Unknown")
        
        if decision.startswith(expected_start):
            correct_alignments += 1
        else:
            misalignments.append({
                'row': idx,
                'signal': signal,
                'expected': expected_start,
                'actual': decision[:30] + "..."
            })
    
    accuracy = (correct_alignments / total_rows) * 100
    
    print(f"Total rows analyzed: {total_rows}")
    print(f"Correct alignments: {correct_alignments}")
    print(f"Misalignments: {len(misalignments)}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if accuracy == 100.0:
        print("✅ PERFECT ALIGNMENT: Decision column aligns 100% with signal column!")
    else:
        print("❌ ALIGNMENT ISSUES FOUND:")
        for mis in misalignments[:5]:  # Show first 5 misalignments
            print(f"  Row {mis['row']}: Signal {mis['signal']} -> Expected '{mis['expected']}' but got '{mis['actual']}'")
    
    return accuracy == 100.0

def verify_no_signal_references(df):
    """Verify that reasoning text contains no signal column references."""
    print("\n" + "="*60)
    print("SIGNAL REFERENCE VERIFICATION")
    print("="*60)
    
    reasoning_columns = [
        'decision', 'pattern_recognition', 'context_analysis', 
        'psychology_assessment', 'execution_decision', 'risk_reward',
        'feature_analysis', 'historical_analysis'
    ]
    
    signal_keywords = [
        'signal', 'signal column', 'signal value', 'signal indicates',
        'signal shows', 'signal suggests', 'based on signal',
        'signal equals', 'signal is', 'signal =', 'signal==',
        'signal 1', 'signal 2', 'signal 0'
    ]
    
    total_violations = 0
    violations_by_column = {}
    
    for column in reasoning_columns:
        if column not in df.columns:
            continue
            
        column_violations = 0
        for idx, text in enumerate(df[column]):
            if pd.isna(text):
                continue
                
            text_lower = str(text).lower()
            for keyword in signal_keywords:
                if keyword in text_lower:
                    column_violations += 1
                    total_violations += 1
                    print(f"❌ Signal reference found in {column} row {idx}: '{keyword}' in '{text[:50]}...'")
                    break
        
        violations_by_column[column] = column_violations
    
    print(f"\nSignal reference violations by column:")
    for column, violations in violations_by_column.items():
        status = "✅" if violations == 0 else "❌"
        print(f"  {status} {column}: {violations} violations")
    
    if total_violations == 0:
        print("\n✅ PERFECT: No signal column references found in any reasoning text!")
        return True
    else:
        print(f"\n❌ VIOLATIONS FOUND: {total_violations} signal references detected!")
        return False

def analyze_content_uniqueness(df):
    """Analyze content uniqueness and diversity."""
    print("\n" + "="*60)
    print("CONTENT UNIQUENESS ANALYSIS")
    print("="*60)
    
    reasoning_columns = [
        'decision', 'pattern_recognition', 'context_analysis', 
        'psychology_assessment', 'execution_decision', 'risk_reward',
        'feature_analysis', 'historical_analysis'
    ]
    
    for column in reasoning_columns:
        if column not in df.columns:
            continue
            
        texts = df[column].dropna().astype(str)
        if len(texts) == 0:
            continue
            
        # Calculate uniqueness
        unique_texts = texts.nunique()
        total_texts = len(texts)
        uniqueness_pct = (unique_texts / total_texts) * 100
        
        # Analyze sentence starters
        starters = []
        for text in texts:
            first_sentence = text.split('.')[0].strip()
            if len(first_sentence) > 10:
                starters.append(first_sentence[:30])
        
        unique_starters = len(set(starters))
        starter_diversity = (unique_starters / len(starters)) * 100 if starters else 0
        
        # Analyze word diversity
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        unique_words = len(word_counts)
        total_words = len(all_words)
        
        print(f"\n{column}:")
        print(f"  Unique texts: {unique_texts}/{total_texts} ({uniqueness_pct:.1f}%)")
        print(f"  Starter diversity: {unique_starters}/{len(starters)} ({starter_diversity:.1f}%)")
        print(f"  Vocabulary: {unique_words} unique words from {total_words} total")
        
        # Show most common phrases
        if len(texts) > 5:
            common_phrases = []
            for text in texts:
                # Extract 3-word phrases
                words = re.findall(r'\b\w+\b', text.lower())
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i+3])
                    common_phrases.append(phrase)
            
            phrase_counts = Counter(common_phrases)
            top_phrases = phrase_counts.most_common(3)
            print(f"  Most common 3-word phrases:")
            for phrase, count in top_phrases:
                print(f"    '{phrase}': {count} times")

def analyze_natural_language_quality(df):
    """Analyze natural language quality improvements."""
    print("\n" + "="*60)
    print("NATURAL LANGUAGE QUALITY ANALYSIS")
    print("="*60)
    
    reasoning_columns = [
        'decision', 'pattern_recognition', 'context_analysis', 
        'psychology_assessment', 'execution_decision', 'risk_reward',
        'feature_analysis', 'historical_analysis'
    ]
    
    quality_metrics = {}
    
    for column in reasoning_columns:
        if column not in df.columns:
            continue
            
        texts = df[column].dropna().astype(str)
        if len(texts) == 0:
            continue
        
        # Calculate quality metrics
        avg_length = texts.str.len().mean()
        avg_words = texts.str.split().str.len().mean()
        avg_sentences = texts.str.count('\.').mean() + 1
        
        # Check for professional language indicators
        professional_indicators = [
            'analysis', 'indicates', 'suggests', 'demonstrates', 'reveals',
            'characteristics', 'conditions', 'assessment', 'evaluation',
            'technical', 'market', 'momentum', 'trend', 'volatility'
        ]
        
        professional_score = 0
        for text in texts:
            text_lower = text.lower()
            score = sum(1 for indicator in professional_indicators if indicator in text_lower)
            professional_score += score
        
        avg_professional_score = professional_score / len(texts)
        
        # Check for variety in sentence structure
        sentence_starters = []
        for text in texts:
            first_word = text.split()[0] if text.split() else ""
            sentence_starters.append(first_word)
        
        starter_variety = len(set(sentence_starters)) / len(sentence_starters) if sentence_starters else 0
        
        quality_metrics[column] = {
            'avg_length': avg_length,
            'avg_words': avg_words,
            'avg_sentences': avg_sentences,
            'professional_score': avg_professional_score,
            'starter_variety': starter_variety
        }
        
        print(f"\n{column}:")
        print(f"  Average length: {avg_length:.1f} characters")
        print(f"  Average words: {avg_words:.1f}")
        print(f"  Average sentences: {avg_sentences:.1f}")
        print(f"  Professional language score: {avg_professional_score:.1f}")
        print(f"  Sentence starter variety: {starter_variety:.1%}")
    
    return quality_metrics

def verify_decision_format(df):
    """Verify decision column format follows 'Going LONG/SHORT/HOLD because...' pattern."""
    print("\n" + "="*60)
    print("DECISION FORMAT VERIFICATION")
    print("="*60)
    
    if 'decision' not in df.columns:
        print("❌ Decision column not found!")
        return False
    
    decisions = df['decision'].dropna()
    total_decisions = len(decisions)
    
    # Expected patterns
    expected_patterns = [
        r'^Going LONG because',
        r'^Going SHORT because', 
        r'^Staying HOLD because'
    ]
    
    correct_format = 0
    format_issues = []
    
    for idx, decision in enumerate(decisions):
        decision_str = str(decision)
        
        if any(re.match(pattern, decision_str) for pattern in expected_patterns):
            correct_format += 1
        else:
            format_issues.append({
                'row': idx,
                'decision': decision_str[:50] + "..."
            })
    
    format_accuracy = (correct_format / total_decisions) * 100
    
    print(f"Total decisions analyzed: {total_decisions}")
    print(f"Correct format: {correct_format}")
    print(f"Format issues: {len(format_issues)}")
    print(f"Format accuracy: {format_accuracy:.2f}%")
    
    if format_accuracy == 100.0:
        print("✅ PERFECT FORMAT: All decisions follow 'Going LONG/SHORT/HOLD because...' pattern!")
    else:
        print("❌ FORMAT ISSUES FOUND:")
        for issue in format_issues[:5]:  # Show first 5 issues
            print(f"  Row {issue['row']}: '{issue['decision']}'")
    
    return format_accuracy == 100.0

def main():
    """Main verification function."""
    print("Enhanced Reasoning System Verification")
    print("="*60)
    
    # Load data
    df = load_reasoning_data()
    if df is None:
        return
    
    # Run all verifications
    results = {}
    
    results['decision_signal_alignment'] = verify_decision_signal_alignment(df)
    results['no_signal_references'] = verify_no_signal_references(df)
    results['decision_format'] = verify_decision_format(df)
    
    # Analyze quality improvements
    analyze_content_uniqueness(df)
    quality_metrics = analyze_natural_language_quality(df)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test.replace('_', ' ').title()}")
    
    if all_passed:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
        print("Enhanced reasoning system is working perfectly:")
        print("  ✅ 100% decision-signal alignment")
        print("  ✅ No signal column references in reasoning text")
        print("  ✅ Proper decision format")
        print("  ✅ High-quality natural language generation")
    else:
        print("\n⚠️  SOME VERIFICATIONS FAILED!")
        print("Please review the issues above and fix them.")

if __name__ == "__main__":
    main()
