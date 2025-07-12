name: "Enhanced Trading Reasoning System with LLM Integration"
description: |

## Purpose

Transform the current template-based reasoning system into an intelligent, diverse, and natural reasoning generator that:
1. Uses signal columns effectively to drive confidence and reasoning
2. Understands feature relationships and market conditions
3. Generates diverse, non-generic reasoning through rule-based structured generation + LLM enhancement
4. Provides signal-aware confidence scoring and profitability insights

## Core Principles

1. **Historical Context Awareness**: Use past market data and patterns to inform current reasoning (like real traders)
2. **No Signal Column Usage**: Never use signal column in reasoning (it's future data/training target)
3. **Feature Understanding**: Leverage all 65 technical indicators intelligently with historical context
4. **Advanced Rule-Based Process**: Sophisticated rule-based reasoning with dynamic templates and natural language patterns
5. **Diversity**: Multiple templates and dynamic selection based on market conditions
6. **Indirect Profitability Hints**: Use market conditions and historical patterns to hint at opportunities
7. **Temporal Reasoning**: Weight both current conditions and historical trends appropriately

---

## Goal

Create a sophisticated reasoning system that generates human-like, diverse trading reasoning that:
- **NEVER uses signal column** (it's future data for training, not available in real trading)
- **Incorporates realistic historical context** - analyzes past 20-50 candles like real traders (short-term), 100-200 candles (medium-term trends)
- **Generates strong profitability reasoning** when all market conditions align favorably
- **Includes decision column** with explicit trading decisions: "Going LONG because...", "Staying HOLD because...", "Going SHORT because..."
- **Uses indirect profitability hints** based on market conditions, momentum, and historical success patterns
- **Understands relationships** between technical indicators across realistic time periods
- **Generates diverse reasoning patterns** (target: 60%+ unique content vs current 4.4%)
- **Uses advanced rule-based templates** with natural language patterns for fast generation
- **Provides temporal insights** combining current conditions with realistic historical analysis periods

## Why

- **Current Issues**: 4.4% unique content, generic templates, no historical context, signal column misuse
- **Data Leakage Risk**: Using signal column in reasoning creates future data leakage
- **Real Trading Simulation**: Traders use historical patterns and context, not just current moment
- **Business Value**: Better training data for ML models with realistic reasoning patterns
- **User Experience**: More natural, insightful reasoning that mimics real trader psychology
- **Model Training**: Historical context helps models learn temporal decision-making patterns

## What

A two-stage reasoning system:

**Advanced Rule-Based Reasoning System**
- **Historical pattern analysis** (past 20-50 and 100-200 periods)
- **Feature relationship understanding** across realistic time periods
- **Market condition detection** with temporal context
- **Trend continuation/reversal analysis** based on historical data
- **Dynamic template selection** based on market conditions and historical patterns
- **Natural language generation** through sophisticated rule-based templates
- **Diverse reasoning patterns** with contextual variations
- **Fast processing** without LLM overhead

### Success Criteria

- [ ] **NEVER uses signal column** in reasoning generation (prevents data leakage)
- [ ] **Realistic historical context integration** - analyzes past 20-50 candles (short-term), 100-200 candles (medium-term)
- [ ] **Decision column implementation** with explicit trading decisions based on market analysis
- [ ] **Strong profitability reasoning** when market conditions align (without mentioning signal column)
- [ ] **Unique content increases** from 4.4% to 60%+ for pattern recognition
- [ ] **Reasoning mentions relevant technical indicators** based on actual values and realistic historical context
- [ ] **Advanced rule-based templates produce natural, varied language** with temporal awareness
- [ ] **Confidence based on market conditions** and historical patterns, not signals
- [ ] **Indirect profitability hints** through market strength analysis and historical success patterns
- [ ] **Processing time remains under 1 second per row** with fast rule-based generation
- [ ] **Temporal reasoning quality** - combines current conditions with realistic trader-like historical analysis

## Decision Column Requirements

### Trading Decision Logic (Without Using Signal Column)

**For LONG Decisions (when signal=1, but don't reference signal):**
```
"Going LONG because:
- Strong bullish momentum confirmed by MACD crossing above signal line
- Price broke above 20-period resistance with volume confirmation
- RSI showing healthy 45-65 range with room for upside
- Historical pattern shows similar setups led to 2-3% moves higher
- All major moving averages aligned bullishly
- Risk-reward ratio favors long position with tight stop below recent support"
```

**For SHORT Decisions (when signal=2, but don't reference signal):**
```
"Going SHORT because:
- Bearish divergence between price and RSI over past 15 candles
- Failed breakout above key resistance level with rejection candle
- MACD histogram showing weakening momentum
- Price below all major moving averages in bearish alignment
- Historical analysis shows similar patterns led to 1-2% declines
- Volume pattern suggests distribution by smart money"
```

**For HOLD Decisions (when signal=0, but don't reference signal):**
```
"Staying HOLD because:
- Mixed signals from technical indicators create uncertainty
- Price consolidating in narrow range between support and resistance
- Low volatility environment with unclear directional bias
- Waiting for clearer confirmation from momentum indicators
- Risk-reward not favorable for either direction currently"
```

### Realistic Historical Analysis Periods

**Real Trader Approach:**
- **Immediate Context (20-50 candles)**: Entry timing, recent support/resistance, short-term momentum
- **Trend Context (100-200 candles)**: Major trend direction, significant levels, pattern development
- **Long-term Context (500+ candles)**: Major market structure, long-term support/resistance (if needed)

**Example Realistic Analysis:**
```
"Looking at the past 30 candles, price has been respecting the ascending trendline with three successful tests.
The broader 150-candle context shows we're in a strong uptrend that began 8 weeks ago.
Recent 20-candle momentum shows acceleration with higher highs and higher lows pattern intact."
```

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- url: https://huggingface.co/microsoft/DialoGPT-small
  why: Small conversational model for text enhancement
  
- url: https://huggingface.co/distilbert-base-uncased
  why: Lightweight model for text generation/enhancement
  
- url: https://huggingface.co/google/flan-t5-small
  why: Instruction-following model for structured text conversion
  
- file: src/reasoning_system/engines/pattern_recognition_engine.py
  why: Current pattern recognition implementation to enhance
  
- file: src/reasoning_system/generators/text_generator.py
  why: Current text generation patterns to improve
  
- file: analyze_features.py
  why: Understanding of 65 feature columns and their categories
  
- file: examine_reasoning.py
  why: Current reasoning quality analysis and issues identified
```

### Current Codebase Analysis

**Feature Categories Available (65 columns):**
- Basic OHLCV: 5 columns (datetime, open, high, low, close)
- Moving Averages: 16 columns (SMA/EMA 5,10,20,50,100,200)
- RSI Indicators: 2 columns (RSI 14, 21)
- MACD: 3 columns (macd, signal, histogram)
- Bollinger Bands: 5 columns (upper, middle, lower, width, position)
- ATR/Volatility: 3 columns
- Market Structure: 7 columns (support/resistance levels, trend data)
- Signals: 1 column (0=HOLD, 1=BUY, 2=SELL)
- Other indicators: 18 columns (ADX, Stochastic, Williams %R, etc.)

**Current Issues:**
- Pattern Recognition: Only 4.4% unique content (35 unique out of 800)
- **CRITICAL**: Signal column used in reasoning (creates data leakage)
- **No historical context** - only looks at current moment (unrealistic)
- **No decision column** - missing explicit trading decisions
- **No strong profitability reasoning** when conditions align
- Generic templates: "standard candle formation without distinct pattern characteristics"
- No feature relationship understanding across realistic time periods
- Confidence scores artificially tied to future signal data
- **Missing temporal reasoning** that real traders use (20-50 candles short-term, 100-200 medium-term)

### Desired Codebase Structure

```bash
src/reasoning_system/
├── engines/
│   ├── __init__.py
│   ├── historical_pattern_engine.py       # Analyzes past 20-50, 100-200 periods for patterns
│   ├── feature_relationship_engine.py     # Understands indicator relationships across time
│   ├── market_condition_detector.py       # Detects trending/ranging/volatile markets with history
│   ├── trend_analysis_engine.py           # Continuation/reversal analysis based on history
│   └── template_selector.py               # Dynamic template selection (NO signal column)
├── advanced_templates/
│   ├── __init__.py
│   ├── natural_language_generator.py      # Advanced rule-based natural language generation
│   ├── template_variations.py             # Multiple variations for each template type
│   ├── contextual_phrases.py              # Context-aware phrase selection
│   └── decision_language_generator.py     # Generates LONG/SHORT/HOLD decision text
├── templates/
│   ├── __init__.py
│   ├── market_condition_templates.py      # Trending/ranging/volatile templates with history
│   ├── historical_pattern_templates.py    # Templates based on historical analysis
│   ├── decision_templates.py              # LONG/SHORT/HOLD decision templates
│   └── feature_templates.py               # Templates based on indicator values and trends
└── enhanced_orchestrator.py               # Fast rule-based orchestrator with historical context
```

### Known Gotchas & Requirements

```python
# CRITICAL: NO SIGNAL COLUMN USAGE
# NEVER use signal column in reasoning (it's future data for training)
# Confidence must come from market conditions and historical patterns
# Never directly mention "profitable" or "will make money"

# CRITICAL: Realistic Historical Context Requirements
# SHORT-TERM: Analyze past 20-50 candles (immediate trend, support/resistance)
# MEDIUM-TERM: Analyze past 100-200 candles (broader trend context, major levels)
# LONG-TERM: Consider 500+ candles for major trend identification (if needed)
# Weight recent data more heavily than distant data (exponential decay)
# Real trader approach: Recent 20 candles for entry timing, 100+ for trend context

# CRITICAL: Feature Understanding with Time
# RSI > 70 = overbought conditions (check if sustained over periods)
# RSI < 30 = oversold conditions (check for divergences)
# MACD > 0 = bullish momentum (analyze trend strength over time)
# Price above SMA = uptrend context (check trend duration and strength)
# Bollinger Band position with historical squeeze/expansion patterns

# CRITICAL: Advanced Rule-Based Generation
# Fast processing without LLM overhead (< 1 second per row)
# Sophisticated template variations for natural language
# Context-aware phrase selection based on market conditions
# Dynamic template selection for diversity

# CRITICAL: Advanced Template Diversity with Historical Context
# Minimum 50+ different pattern recognition templates with variations
# Historical pattern-based template selection (20-50, 100-200 candles)
# Feature value-based template variations with realistic time context
# Market condition intensity based on historical analysis
# Natural language variations for same concepts (5-10 ways to say same thing)
# Contextual phrase selection based on market strength/weakness
```

## Implementation Blueprint

### Data Models and Structure

```python
# Enhanced reasoning data structures
@dataclass
class HistoricalContext:
    short_term_lookback: int  # 20-50 candles (immediate trend analysis)
    medium_term_lookback: int  # 100-200 candles (broader trend context)
    trend_duration: int    # How long current trend has lasted
    trend_strength: str    # "weak", "moderate", "strong" based on consistency
    pattern_type: str      # "continuation", "reversal", "consolidation"
    volatility_trend: str  # "increasing", "decreasing", "stable"
    momentum_history: str  # "accelerating", "decelerating", "consistent"
    support_resistance_history: str  # Historical S/R level tests and breaks
    volume_pattern: str    # "increasing", "decreasing", "diverging" (if volume available)

@dataclass
class MarketCondition:
    current_trend_direction: str  # "bullish", "bearish", "sideways"
    historical_trend_strength: str  # Based on past periods
    volatility_level: str  # "low", "medium", "high"
    momentum_strength: str  # "weak", "moderate", "strong"
    support_resistance_context: str
    historical_context: HistoricalContext

@dataclass
class FeatureAnalysis:
    rsi_condition: str  # "oversold", "neutral", "overbought"
    rsi_divergence: str  # "bullish", "bearish", "none" (vs price history)
    macd_signal: str   # "bullish", "bearish", "neutral"
    macd_trend: str    # Historical MACD trend analysis
    ma_alignment: str  # "bullish", "bearish", "mixed"
    ma_trend_strength: str  # Based on historical MA relationships
    bb_position: str   # "lower", "middle", "upper"
    bb_squeeze_expansion: str  # Historical volatility pattern

@dataclass
class TradingDecision:
    decision_type: str  # "LONG", "SHORT", "HOLD"
    decision_reasoning: str  # Detailed explanation why this decision was made
    market_alignment_score: float  # How well conditions align (0-100)
    risk_factors: List[str]  # Potential risks identified
    profit_potential_hints: List[str]  # Indirect profitability indicators

@dataclass
class StructuredReasoning:
    market_condition: MarketCondition
    feature_analysis: FeatureAnalysis
    historical_context: HistoricalContext
    trading_decision: TradingDecision
    template_category: str
    key_indicators: List[str]
    confidence_factors: List[str]  # Based on market conditions, NOT signals
```

### Implementation Tasks

```yaml
Task 1: Create Realistic Historical Pattern Analysis Engine
CREATE src/reasoning_system/engines/historical_pattern_engine.py:
  - ANALYZE past 20-50 candles for immediate trend patterns and entry timing
  - ANALYZE past 100-200 candles for broader trend context and major levels
  - DETECT pattern reversals and continuations like real traders
  - WEIGHT recent data more heavily than distant data (exponential decay)
  - IDENTIFY historical volatility and momentum patterns
  - ANALYZE support/resistance level tests and breaks over time
  - NEVER use signal column (future data)

Task 2: Build Enhanced Feature Analysis Engine
CREATE src/reasoning_system/engines/feature_relationship_engine.py:
  - ANALYZE all 65 feature columns with historical context
  - DETECT RSI overbought/oversold with divergence analysis
  - IDENTIFY MACD bullish/bearish signals with trend strength
  - DETERMINE moving average alignment and historical strength
  - ASSESS Bollinger Band position with squeeze/expansion history

Task 3: Create Market Condition Detection with History
CREATE src/reasoning_system/engines/market_condition_detector.py:
  - DETECT trending vs ranging markets with historical duration
  - IDENTIFY volatility levels and trends over time
  - DETERMINE momentum strength and acceleration/deceleration
  - CLASSIFY market regime based on historical patterns
  - GENERATE confidence based on market conditions, NOT signals

Task 4: Build Decision Column and Template System
CREATE src/reasoning_system/templates/decision_templates.py:
  - CREATE decision templates for LONG positions: "Going LONG because market shows..."
  - CREATE decision templates for SHORT positions: "Going SHORT because trend indicates..."
  - CREATE decision templates for HOLD positions: "Staying HOLD because conditions are..."
  - INCLUDE strong profitability reasoning when all conditions align
  - NEVER reference signal column, use market condition alignment instead
  - Go with long decision only when the signal column is 1.
  - Go with short decision only when the signal column is 2.
  - Hold decision only when the signal column is 0.

CREATE src/reasoning_system/templates/historical_pattern_templates.py:
  - CREATE 20+ diverse templates for trend continuation patterns
  - CREATE 20+ diverse templates for reversal patterns
  - CREATE 15+ diverse templates for consolidation/ranging markets
  - IMPLEMENT template selection based on realistic historical analysis
  - INCLUDE indirect profitability hints through market strength and alignment

Task 5: Build Advanced Natural Language Generation
CREATE src/reasoning_system/advanced_templates/natural_language_generator.py:
  - CREATE sophisticated rule-based natural language patterns
  - IMPLEMENT multiple variations for same concepts (5-10 ways to express each idea)
  - GENERATE contextual phrases based on market strength/weakness
  - MAINTAIN technical accuracy while adding natural variety

Task 6: Create Template Variation System
CREATE src/reasoning_system/advanced_templates/template_variations.py:
  - BUILD 50+ different pattern recognition templates with variations
  - CREATE contextual phrase selection based on market conditions
  - IMPLEMENT dynamic language intensity based on signal strength (without using signal)
  - GENERATE diverse expressions for historical patterns

Task 7: Build Enhanced Orchestrator with Fast Processing
CREATE src/reasoning_system/enhanced_orchestrator.py:
  - IMPLEMENT fast rule-based process: historical analysis → feature analysis → template selection → natural language generation
  - COORDINATE all engines for comprehensive reasoning generation
  - ENSURE confidence based on market conditions and historical patterns
  - MAINTAIN fast processing speed (< 1 second per row)
  - NEVER use signal column in any reasoning component
```

### Integration Points

```yaml
CONFIGURATION:
  - add to: src/config/reasoning_config.py
  - settings: Template diversity settings, historical analysis periods, confidence ranges

DEPENDENCIES:
  - NO additional packages needed (pure rule-based system)
  - existing: pandas, numpy for data processing

PIPELINE:
  - modify: src/data_processing/reasoning_processor.py
  - change: Use enhanced_orchestrator instead of current orchestrator
  - benefit: Much faster processing without LLM overhead
```

## Validation Loop

### Level 1: Component Testing

```bash
# Test historical pattern analysis
python -c "
from src.reasoning_system.engines.historical_pattern_engine import HistoricalAnalyzer
analyzer = HistoricalAnalyzer()
# Test with sample data (past 20 periods)
result = analyzer.analyze_historical_patterns(sample_data_20_periods)
assert result.pattern_type in ['continuation', 'reversal', 'consolidation']
assert result.trend_duration > 0
assert 'signal' not in str(result)  # Ensure no signal column usage
"

# Test template diversity with historical context
python -c "
from src.reasoning_system.templates.historical_pattern_templates import HistoricalTemplates
templates = HistoricalTemplates()
continuation_templates = templates.get_continuation_templates()
assert len(continuation_templates) >= 20
assert len(set(continuation_templates)) == len(continuation_templates)  # All unique
"
```

### Level 2: Integration Testing

```bash
# Test enhanced reasoning generation with historical context
python -c "
import pandas as pd
from src.reasoning_system.enhanced_orchestrator import EnhancedReasoningOrchestrator

df = pd.read_csv('data/processed/features_Bank_Nifty_5.csv').head(50)  # Need more data for historical analysis
orchestrator = EnhancedReasoningOrchestrator()
result = orchestrator.process_dataframe(df)

# Verify NO signal column usage in reasoning
reasoning_text = ' '.join([
    str(result['pattern_recognition_text'].iloc[0]),
    str(result['context_analysis_text'].iloc[0]),
    str(result['psychology_assessment_text'].iloc[0])
])
assert 'signal' not in reasoning_text.lower()  # CRITICAL: No signal mentions

# Verify historical context integration
assert 'trend' in reasoning_text.lower() or 'momentum' in reasoning_text.lower()
assert 'historical' in reasoning_text.lower() or 'past' in reasoning_text.lower()

# Verify confidence based on market conditions, not signals
strong_trend_rows = result[result['trend_strength'] > 0.7]  # Assuming trend strength feature
assert strong_trend_rows['confidence_score'].mean() > 70  # Strong trends should have higher confidence
"
```

### Level 3: Quality Validation

```bash
# Test reasoning diversity
python scripts/test_reasoning_diversity.py
# Expected: >60% unique content for pattern recognition
# Expected: Signal-appropriate confidence levels
# Expected: Relevant technical indicators mentioned

# Test advanced rule-based generation
python scripts/test_natural_language_generation.py
# Expected: Natural, varied language output
# Expected: Preserved technical accuracy
# Expected: Processing time < 1 second per row
```

## Final Validation Checklist

- [ ] **CRITICAL**: Signal column NEVER used in reasoning (prevents data leakage)
- [ ] **Decision column implemented**: Explicit "Going LONG/SHORT/HOLD because..." decisions
- [ ] **Strong profitability reasoning**: When all market conditions align favorably
- [ ] **Realistic historical context**: Past 20-50 candles (short-term), 100-200 candles (medium-term)
- [ ] **Pattern recognition uniqueness** > 60% (vs current 4.4%)
- [ ] **Reasoning mentions relevant indicators** based on actual values and realistic historical trends
- [ ] **LLM produces natural, varied language** with temporal awareness
- [ ] **Confidence based on market conditions** and historical patterns, NOT signals
- [ ] **Indirect profitability hints** through market strength and historical success patterns
- [ ] **No direct profitability mentions** ("profitable", "will make money", etc.)
- [ ] **Processing speed is fast** (< 1 second per row) with rule-based generation
- [ ] **Temporal reasoning quality**: Combines current conditions with realistic trader-like historical analysis
- [ ] **Decision reasoning quality**: Clear explanations for LONG/SHORT/HOLD decisions
- [ ] **All tests pass**: `python -m pytest tests/reasoning/ -v`
- [ ] **Integration test with full pipeline successful**
- [ ] **Data leakage verification**: No future information used in reasoning

---

## Anti-Patterns to Avoid

- ❌ **NEVER use signal column in reasoning** (it's future data - creates data leakage)
- ❌ Don't directly mention "profitable", "will make money", or specific returns
- ❌ Don't ignore historical context - reasoning must consider past patterns
- ❌ Don't use generic templates that could apply to any market condition
- ❌ Don't let LLM hallucinate technical indicator values
- ❌ Don't sacrifice processing speed for marginal quality improvements
- ❌ Don't make confidence scores based on future signal data
- ❌ Don't analyze only current moment - real traders use historical context
- ❌ Don't create reasoning that couldn't be generated in real-time trading

## Summary

This PRP provides a comprehensive roadmap to transform the current template-based reasoning system into an intelligent, diverse, and natural reasoning generator. The two-stage approach (rule-based structured generation + LLM enhancement) will address all current issues while maintaining processing efficiency.

**Key Improvements:**
1. **Historical Context**: Analyzes past 20-50, 100-200 periods like real traders do
2. **No Data Leakage**: NEVER uses signal column (future data) in reasoning
3. **Feature-Aware**: Understands relationships between 65 technical indicators across time
4. **Diverse Templates**: 60%+ unique content vs current 4.4% through 50+ template variations
5. **Fast Rule-Based Generation**: Advanced natural language without LLM overhead (< 1 second per row)
6. **Market-Aware**: Dynamic template selection based on market conditions and historical patterns
7. **Decision Column**: Explicit LONG/SHORT/HOLD decisions with detailed reasoning
8. **Indirect Profitability**: Hints at opportunities through market strength, not direct predictions

**Expected Outcome:** High-quality, diverse reasoning data perfect for ML model training with natural language patterns that reflect real trader psychology and temporal decision-making processes. Fast generation enables processing large datasets efficiently while maintaining quality. The model will learn to make decisions based on historical context and market conditions, just like human traders.
