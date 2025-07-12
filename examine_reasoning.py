#!/usr/bin/env python3
"""
Examine Reasoning Quality
========================

Script to examine the actual reasoning content and validate its quality.
"""

import pandas as pd
import numpy as np

def examine_reasoning():
    """Examine the reasoning content in detail."""
    
    # Load the final data
    df = pd.read_csv('data/final/final_features_Bank_Nifty_5.csv')
    
    print("="*80)
    print("REASONING QUALITY EXAMINATION")
    print("="*80)
    
    print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Find reasoning columns
    reasoning_cols = [col for col in df.columns if 'text' in col or 'confidence' in col]
    print(f"\nReasoning columns found: {len(reasoning_cols)}")
    for col in reasoning_cols:
        print(f"  - {col}")
    
    # Examine sample reasoning from different rows
    sample_rows = [0, len(df)//4, len(df)//2, len(df)-1]  # First, quarter, middle, last
    
    for i, row_idx in enumerate(sample_rows):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i+1}: ROW {row_idx}")
        print(f"{'='*60}")
        
        row = df.iloc[row_idx]
        
        # Show market data context
        print(f"Market Context:")
        print(f"  DateTime: {row['datetime']}")
        print(f"  OHLC: O={row['open']:.2f}, H={row['high']:.2f}, L={row['low']:.2f}, C={row['close']:.2f}")
        if 'signal' in df.columns:
            signal_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            print(f"  Signal: {signal_map.get(row.get('signal', 0), 'UNKNOWN')}")
        
        # Show reasoning content
        print(f"\nReasoning Analysis:")
        
        for col in reasoning_cols:
            if col in row and pd.notna(row[col]):
                content = str(row[col])
                print(f"\n{col.replace('_', ' ').title()}:")
                if len(content) > 300:
                    print(f"  {content[:300]}...")
                    print(f"  [Content length: {len(content)} characters]")
                else:
                    print(f"  {content}")
            else:
                print(f"\n{col.replace('_', ' ').title()}: [MISSING OR NULL]")
    
    # Quality analysis
    print(f"\n{'='*80}")
    print("QUALITY ANALYSIS")
    print(f"{'='*80}")
    
    if 'confidence_score' in df.columns:
        conf_scores = df['confidence_score'].dropna()
        print(f"Confidence Scores:")
        print(f"  Mean: {conf_scores.mean():.1f}")
        print(f"  Std: {conf_scores.std():.1f}")
        print(f"  Range: {conf_scores.min():.1f} - {conf_scores.max():.1f}")
        print(f"  Distribution:")
        print(f"    High (80-100): {sum(conf_scores >= 80)} rows ({sum(conf_scores >= 80)/len(conf_scores)*100:.1f}%)")
        print(f"    Medium (60-80): {sum((conf_scores >= 60) & (conf_scores < 80))} rows")
        print(f"    Low (0-60): {sum(conf_scores < 60)} rows")
    
    # Check for common issues
    print(f"\nContent Quality Checks:")
    
    text_columns = [col for col in reasoning_cols if 'text' in col]
    
    for col in text_columns:
        if col in df.columns:
            texts = df[col].dropna().astype(str)
            
            # Check for empty or very short content
            empty_count = sum(texts.str.len() < 10)
            
            # Check for repetitive content
            unique_count = texts.nunique()
            
            # Check for price references (should be avoided)
            price_refs = sum(texts.str.contains(r'\$\d+|\d+\.\d+\s*(dollars|USD)', case=False, na=False))
            
            print(f"\n  {col.replace('_', ' ').title()}:")
            print(f"    Total entries: {len(texts)}")
            print(f"    Unique content: {unique_count} ({unique_count/len(texts)*100:.1f}%)")
            print(f"    Empty/short (<10 chars): {empty_count}")
            print(f"    Price references: {price_refs}")
            print(f"    Avg length: {texts.str.len().mean():.0f} characters")

def check_reasoning_validity():
    """Check if reasoning makes logical sense."""
    
    df = pd.read_csv('data/final/final_features_Bank_Nifty_5.csv')
    
    print(f"\n{'='*80}")
    print("REASONING VALIDITY CHECK")
    print(f"{'='*80}")
    
    # Check a few specific examples for logical consistency
    sample_indices = [0, 100, 200, 300] if len(df) > 300 else [0, len(df)//3, 2*len(df)//3]
    
    for idx in sample_indices:
        if idx >= len(df):
            continue
            
        row = df.iloc[idx]
        
        print(f"\nRow {idx} Validity Check:")
        print(f"Market: O={row['open']:.2f}, H={row['high']:.2f}, L={row['low']:.2f}, C={row['close']:.2f}")
        
        # Check if pattern recognition mentions relevant technical indicators
        if 'pattern_recognition_text' in row:
            pattern_text = str(row['pattern_recognition_text']).lower()
            
            # Check if it mentions actual technical indicators from the data
            indicators_mentioned = []
            if 'moving average' in pattern_text or 'ma' in pattern_text:
                indicators_mentioned.append('Moving Averages')
            if 'rsi' in pattern_text:
                indicators_mentioned.append('RSI')
            if 'macd' in pattern_text:
                indicators_mentioned.append('MACD')
            if 'bollinger' in pattern_text:
                indicators_mentioned.append('Bollinger Bands')
            
            print(f"  Technical indicators mentioned: {indicators_mentioned}")
            
        # Check if confidence score aligns with signal strength
        if 'confidence_score' in row and 'signal' in row:
            confidence = row['confidence_score']
            signal = row['signal']
            
            print(f"  Signal: {signal}, Confidence: {confidence}")
            
            # High confidence should align with clear signals (not hold)
            if confidence > 80 and signal == 0:  # High confidence but HOLD signal
                print(f"  ⚠️  Potential inconsistency: High confidence ({confidence}) with HOLD signal")
            elif confidence < 60 and signal != 0:  # Low confidence but clear signal
                print(f"  ⚠️  Potential inconsistency: Low confidence ({confidence}) with clear signal")
            else:
                print(f"  ✓ Confidence and signal appear consistent")

if __name__ == "__main__":
    examine_reasoning()
    check_reasoning_validity()
