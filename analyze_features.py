#!/usr/bin/env python3
"""
Analyze Features for Reasoning Improvement
==========================================

Analyze the feature columns to understand what data we have for better reasoning.
"""

import pandas as pd
import numpy as np

def analyze_features():
    """Analyze the feature columns in detail."""
    
    df = pd.read_csv('data/final/final_features_Bank_Nifty_5.csv')
    
    print("="*80)
    print("FEATURE ANALYSIS FOR REASONING IMPROVEMENT")
    print("="*80)
    
    # Identify reasoning vs feature columns
    reasoning_cols = [
        'pattern_recognition_text', 'context_analysis_text', 
        'psychology_assessment_text', 'execution_decision_text', 
        'confidence_score', 'risk_assessment_text', 'alternative_scenarios_text'
    ]
    
    feature_cols = [col for col in df.columns if col not in reasoning_cols]
    
    print(f"Total columns: {len(df.columns)}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Reasoning columns: {len(reasoning_cols)}")
    
    # Categorize feature columns
    print(f"\n{'='*60}")
    print("FEATURE CATEGORIES")
    print(f"{'='*60}")
    
    categories = {
        'Basic OHLCV': [],
        'Moving Averages': [],
        'RSI Indicators': [],
        'MACD': [],
        'Bollinger Bands': [],
        'ATR/Volatility': [],
        'Stochastic': [],
        'Williams %R': [],
        'CCI': [],
        'ROC': [],
        'Market Structure': [],
        'Signals': [],
        'Other': []
    }
    
    for col in feature_cols:
        col_lower = col.lower()
        if col in ['datetime', 'open', 'high', 'low', 'close', 'volume']:
            categories['Basic OHLCV'].append(col)
        elif 'sma' in col_lower or 'ema' in col_lower:
            categories['Moving Averages'].append(col)
        elif 'rsi' in col_lower:
            categories['RSI Indicators'].append(col)
        elif 'macd' in col_lower:
            categories['MACD'].append(col)
        elif 'bb_' in col_lower or 'bollinger' in col_lower:
            categories['Bollinger Bands'].append(col)
        elif 'atr' in col_lower or 'volatility' in col_lower:
            categories['ATR/Volatility'].append(col)
        elif 'stoch' in col_lower:
            categories['Stochastic'].append(col)
        elif 'williams' in col_lower or 'wr' in col_lower:
            categories['Williams %R'].append(col)
        elif 'cci' in col_lower:
            categories['CCI'].append(col)
        elif 'roc' in col_lower:
            categories['ROC'].append(col)
        elif any(x in col_lower for x in ['support', 'resistance', 'trend', 'structure']):
            categories['Market Structure'].append(col)
        elif 'signal' in col_lower:
            categories['Signals'].append(col)
        else:
            categories['Other'].append(col)
    
    for category, cols in categories.items():
        if cols:
            print(f"\n{category} ({len(cols)} columns):")
            for col in cols[:10]:  # Show first 10
                print(f"  - {col}")
            if len(cols) > 10:
                print(f"  ... and {len(cols) - 10} more")
    
    # Analyze signal distribution
    print(f"\n{'='*60}")
    print("SIGNAL ANALYSIS")
    print(f"{'='*60}")
    
    if 'signal' in df.columns:
        signal_counts = df['signal'].value_counts()
        signal_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        
        print("Signal Distribution:")
        for signal, count in signal_counts.items():
            signal_name = signal_map.get(signal, f'UNKNOWN({signal})')
            percentage = count / len(df) * 100
            print(f"  {signal_name}: {count} ({percentage:.1f}%)")
    
    # Sample data analysis
    print(f"\n{'='*60}")
    print("SAMPLE DATA ANALYSIS")
    print(f"{'='*60}")
    
    sample_row = df.iloc[100]  # Middle row
    
    print(f"Sample Row Analysis (Row 100):")
    print(f"DateTime: {sample_row['datetime']}")
    print(f"OHLC: O={sample_row['open']:.2f}, H={sample_row['high']:.2f}, L={sample_row['low']:.2f}, C={sample_row['close']:.2f}")
    
    if 'signal' in df.columns:
        signal_name = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(sample_row['signal'], 'UNKNOWN')
        print(f"Signal: {signal_name}")
    
    # Show some key indicators
    key_indicators = ['sma_20', 'ema_20', 'rsi_14', 'macd', 'bb_upper', 'bb_lower', 'atr_14']
    print(f"\nKey Technical Indicators:")
    for indicator in key_indicators:
        if indicator in df.columns:
            value = sample_row[indicator]
            print(f"  {indicator}: {value:.4f}")
    
    return categories, df

def analyze_reasoning_issues():
    """Analyze current reasoning issues."""
    
    print(f"\n{'='*80}")
    print("CURRENT REASONING ISSUES ANALYSIS")
    print(f"{'='*80}")
    
    print("Issues identified:")
    print("1. ❌ Signal column not used effectively in reasoning")
    print("2. ❌ Generic templates with low diversity (4.4% unique pattern recognition)")
    print("3. ❌ No direct mention of profitability or signal strength")
    print("4. ❌ Limited understanding of feature relationships")
    print("5. ❌ No dynamic template selection based on market conditions")
    print("6. ❌ Confidence scores don't reflect signal conviction properly")
    
    print(f"\nCurrent reasoning approach:")
    print("- Uses rule-based templates")
    print("- Limited feature understanding")
    print("- Generic language patterns")
    print("- No LLM enhancement")
    
    print(f"\nProposed improvements:")
    print("1. ✅ Use signal column to drive reasoning confidence")
    print("2. ✅ Create diverse templates based on market conditions")
    print("3. ✅ Better feature understanding and relationships")
    print("4. ✅ Two-stage approach: Rule-based → LLM enhancement")
    print("5. ✅ Dynamic template selection")
    print("6. ✅ Signal-aware confidence scoring")

if __name__ == "__main__":
    categories, df = analyze_features()
    analyze_reasoning_issues()
