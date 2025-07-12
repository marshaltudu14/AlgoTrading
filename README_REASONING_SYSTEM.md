# Automated Trading Reasoning Generation System

A sophisticated system that generates human-like trading reasoning for each row of processed market data. Creates 7 reasoning columns that simulate professional trader thinking patterns using historical context and technical analysis.

## 🎯 Overview

This system transforms raw technical indicators into comprehensive, human-readable trading reasoning that includes:

- **Pattern Recognition**: Candlestick and chart pattern analysis
- **Context Analysis**: Market structure and technical confluence
- **Psychology Assessment**: Market sentiment and participant behavior
- **Execution Decisions**: Trading recommendations with risk-reward analysis
- **Risk Assessment**: Comprehensive risk analysis and scenario planning
- **Alternative Scenarios**: Different market interpretations and contingencies
- **Confidence Scoring**: Numerical confidence based on technical confluence

## 🏗️ Architecture

### Modular Design
```
reasoning_system/
├── core/                    # Core orchestration and base classes
│   ├── reasoning_orchestrator.py
│   └── base_engine.py
├── context/                 # Historical context management
│   └── historical_context_manager.py
├── engines/                 # Specialized reasoning engines
│   ├── pattern_recognition_engine.py
│   ├── context_analysis_engine.py
│   ├── psychology_assessment_engine.py
│   ├── execution_decision_engine.py
│   └── risk_assessment_engine.py
└── generators/              # Text generation and quality validation
    ├── text_generator.py
    └── quality_validator.py
```

### Key Components

1. **ReasoningOrchestrator**: Main coordinator that manages all reasoning engines
2. **HistoricalContextManager**: Analyzes previous 100 candles for context
3. **Reasoning Engines**: Specialized engines for different types of analysis
4. **TextGenerator**: Ensures professional trading language
5. **QualityValidator**: Validates reasoning quality and consistency

## 📊 Generated Reasoning Columns

| Column | Description | Example |
|--------|-------------|---------|
| `pattern_recognition_text` | Candlestick and chart pattern analysis | "Current hammer formation at key support zone shows strong rejection characteristics..." |
| `context_analysis_text` | Market structure and technical confluence | "Market structure reveals consolidation characteristics with neutral directional bias..." |
| `psychology_assessment_text` | Market sentiment and participant behavior | "Market participants demonstrate fear-driven behavior with defensive positioning..." |
| `execution_decision_text` | Trading recommendations and positioning | "Technical setup demonstrates good quality with favorable risk-reward at 2.1:1 ratio..." |
| `confidence_score` | Numerical confidence (0-100) | 75 |
| `risk_assessment_text` | Risk analysis and scenario planning | "Current risk environment shows manageable volatility conditions..." |
| `alternative_scenarios_text` | Alternative interpretations and scenarios | "Trend continuation scenario assumes sustained bullish momentum..." |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, pandas-ta
- Processed feature data from `feature_generator.py`

### Installation
```bash
# No additional installation required - uses existing project structure
# Ensure you have the required dependencies from requirements.txt
```

### Basic Usage
```bash
# Process all feature files in processed_data folder
python reasoning_processor.py

# Specify custom input/output directories
python reasoning_processor.py --input-dir processed_data --output-dir reasoning_data

# Generate detailed quality reports
python reasoning_processor.py --quality-reports
```

### Advanced Usage
```python
from reasoning_system import ReasoningOrchestrator
from reasoning_config import get_reasoning_config

# Initialize with custom configuration
config = get_reasoning_config()
orchestrator = ReasoningOrchestrator(config)

# Process single file
result = orchestrator.process_file('input.csv', 'output.csv')

# Process entire directory
summary = orchestrator.process_directory('input_dir', 'output_dir')
```

## ⚙️ Configuration

### Main Configuration Files
- `reasoning_config.py`: Comprehensive reasoning system configuration
- `config.py`: Base system configuration (from feature generator)

### Key Configuration Options

```python
REASONING_CONFIG = {
    'context_window_size': 100,  # Historical context window
    'pattern_recognition': {
        'support_resistance_threshold': 1.5,
        'pattern_confidence_threshold': 0.7,
    },
    'quality_validation': {
        'min_quality_score': 70,
        'check_price_references': True,
    }
}
```

## 🎨 Key Features

### 1. Scale-Agnostic Reasoning
- **No Absolute Prices**: Uses relative terms like "key support zone" instead of specific price levels
- **Relative Language**: "Current resistance level" vs "23,450 resistance"
- **Universal Application**: Works across different instruments and timeframes

### 2. Historical Context Integration
- **100-Candle Window**: Analyzes previous 100 candles for context
- **Pattern Frequency**: Tracks bullish/bearish pattern bias
- **Trend Consistency**: Evaluates trend strength and persistence
- **Volatility Analysis**: Historical volatility patterns and trends

### 3. Professional Trading Language
- **Authentic Terminology**: Uses real trading language and phrases
- **Logical Flow**: Ensures coherent reasoning progression
- **Confidence Expression**: Natural ways to express uncertainty and conviction
- **Actionable Insights**: Each reasoning leads to clear conclusions

### 4. Quality Assurance
- **Multi-Layer Validation**: Text structure, professional language, consistency
- **Price Reference Detection**: Automatically detects and prevents absolute price mentions
- **Cross-Column Consistency**: Ensures reasoning coherence across all columns
- **Quality Scoring**: Comprehensive quality metrics (0-100)

### 5. Independent Timeframe Analysis
- **No Cross-References**: Each timeframe analyzed independently
- **Pure Technical Analysis**: Based solely on current timeframe data
- **Scalable Design**: Easy to add new timeframes or instruments

## 📈 Example Output

### Input (Feature Data)
```csv
datetime,open,high,low,close,rsi_14,macd_histogram,trend_strength,signal,...
2024-03-18 12:34:00,46653.45,46669.75,46613.75,46618.3,50.87,-14.89,0.28,0,...
```

### Output (With Reasoning)
```csv
datetime,open,high,low,close,...,pattern_recognition_text,context_analysis_text,psychology_assessment_text,execution_decision_text,confidence_score,risk_assessment_text,alternative_scenarios_text
2024-03-18 12:34:00,46653.45,46669.75,46613.75,46618.3,...,"Current candle formation displays standard characteristics without distinct pattern features. Price action remains within normal parameters positioned near key support zone with moderate historical significance. Recent pattern analysis reveals neutral bias over the observation period suggesting balanced market conditions.","Market structure reveals consolidation characteristics with neutral directional bias and weak momentum characteristics. Technical indicators present mixed signals requiring careful interpretation. Momentum environment remains balanced with low directional consistency. Current volatility environment registers normal levels with stable characteristics.","Market participants demonstrate balanced sentiment with no clear emotional extremes based on momentum indicators. Current price action reflects normal trading behavior without significant conviction. Crowd behavior indicates neutral bias in recent positioning with standard risk appetite and positioning characteristics.","Technical setup demonstrates moderate quality with confluence score of 0.4 presenting acceptable risk-reward profile supporting selective positioning. Current analysis supports cautious approach with moderate conviction recommending selective positioning with tight risk management. Market conditions suggest patience until clearer directional signals develop.",65,"Current risk environment shows manageable volatility conditions with standard risk parameters. Market structure presents range-bound structure with breakout risk considerations. Primary alternative scenario suggests continuation of current range-bound behavior with limited directional movement while secondary consideration includes breakout scenario anticipates significant movement beyond current ranges.","Trend continuation scenario expects range-bound behavior with defined boundaries. Volatility expansion scenario could trigger rapid price movement beyond current expectations. Regime change scenario could establish new trending phase with sustained directional movement."
```

## 🔧 Customization

### Adding New Reasoning Engines
1. Inherit from `BaseReasoningEngine`
2. Implement `generate_reasoning()` method
3. Add to `ReasoningOrchestrator`

```python
class CustomReasoningEngine(BaseReasoningEngine):
    def generate_reasoning(self, current_data, context):
        # Your custom reasoning logic
        return "Custom reasoning text..."
```

### Modifying Text Generation
- Edit templates in individual engines
- Customize professional language in `TextGenerator`
- Adjust quality validation rules in `QualityValidator`

### Configuration Customization
- Modify thresholds in `reasoning_config.py`
- Add new validation rules
- Customize text length requirements

## 📊 Quality Metrics

### Quality Scoring (0-100)
- **Professional Language**: 25% weight
- **Text Structure**: 20% weight
- **Price Reference Compliance**: 20% weight
- **Logical Consistency**: 20% weight
- **Required Patterns**: 15% weight

### Quality Thresholds
- **Excellent**: 95+ (Publication ready)
- **Good**: 85+ (Training ready)
- **Acceptable**: 70+ (Usable with review)
- **Poor**: <70 (Needs improvement)

## 🐛 Troubleshooting

### Common Issues

1. **Missing Feature Columns**
   - Ensure feature files contain all required technical indicators
   - Check column names match expected format

2. **Low Quality Scores**
   - Review reasoning templates and logic
   - Check for absolute price references
   - Validate cross-column consistency

3. **Memory Issues**
   - Reduce batch size in configuration
   - Enable memory optimization
   - Process files individually

4. **Performance Issues**
   - Enable context caching
   - Reduce context window size
   - Use batch processing

### Debug Mode
```bash
# Enable detailed logging
python reasoning_processor.py --debug

# Generate quality reports for analysis
python reasoning_processor.py --quality-reports
```

## 🔮 Future Enhancements

- **Multi-Language Support**: Generate reasoning in multiple languages
- **Custom Templates**: User-defined reasoning templates
- **Real-Time Processing**: Stream processing capabilities
- **Advanced NLP**: Integration with large language models
- **Backtesting Integration**: Direct integration with backtesting systems

## 📝 License

This reasoning system is part of the AlgoTrading project and follows the same licensing terms.

## 🤝 Contributing

1. Follow the modular architecture
2. Maintain professional trading language standards
3. Ensure comprehensive testing
4. Update documentation for new features

---

**Note**: This system generates reasoning based on technical analysis only. It does not provide financial advice and should be used for educational and research purposes.
