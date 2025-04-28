RL# Advanced Algorithmic Trading Bot with RLHF and MoE

This project implements an advanced algorithmic trading bot using state-of-the-art reinforcement learning techniques with human feedback (RLHF) and Mixture of Experts (MoE) architecture. The bot is designed to trade index options based on market data and position information, with a focus on Indian market indices.

## Features

- **Mixture of Experts (MoE) Architecture**: Specialized expert networks for different market regimes and patterns
- **Custom PyTorch Transformer Model**: Implements a Decision Transformer architecture with market regime detection and task embedding
- **RLHF Pipeline**: Three-stage training with behavioral cloning, reward modeling, and RL fine-tuning
- **Meta-Learning**: Trains across multiple instruments and timeframes for better generalization
- **Position-Aware**: Incorporates position information into the model for better decision making
- **Risk Management**: Includes risk assessment to avoid high-risk trades
- **Enhanced Reward Shaping**: Uses signal labels for RLHF without exposing future data during inference
- **Anti-Overfitting Measures**: Ensemble models, regularization, and cross-validation
- **Enhanced Feature Engineering**: Advanced technical indicators, market regime detection, and pattern recognition
- **Indian Market Adaptation**: Proper handling of Indian index instruments and their specific quantities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/algotrading.git
cd algotrading

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- `models/`: Contains the model architecture and training code
  - `trading_transformer.py`: Decision Transformer model architecture
  - `moe_transformer.py`: Mixture of Experts model architecture
  - `datasets.py`: Dataset classes for training
  - `training.py`: Training pipeline
  - `inference.py`: Inference code for live trading
- `envs/`: Contains the trading environment
  - `trading_env.py`: Enhanced Gymnasium environment for trading
  - `data_fetcher.py`: Fetches market data
- `data_processing/`: Contains data processing code
  - `processor.py`: Basic data processing
  - `enhanced_processor.py`: Advanced feature engineering
- `feature_engine.py`: Builds features for the model
- `live_trading.py`: Runs the model in live trading mode
- `train_model.py`: Script to train the Decision Transformer model
- `train_moe_model.py`: Script to train the Mixture of Experts model
- `evaluate_transformer.py`: Script to evaluate the Decision Transformer model
- `evaluate_moe_model.py`: Script to evaluate the Mixture of Experts model
- `process_enhanced_data.py`: Script to process data with enhanced features
- `cleanup.py`: Script to remove deprecated files

## Processing Data with Enhanced Features

To process data with enhanced features, run:

```bash
python process_enhanced_data.py --input_dir historical_data --output_dir processed_data --normalize
```

This will process all CSV files in the input directory with enhanced features and save them to the output directory.

## Training the Models

### Training the Decision Transformer Model

To train the Decision Transformer model, run:

```bash
python train_model.py --instruments Nifty "Bank Nifty" --timeframes 5 15 --bc_epochs 50 --rm_epochs 30 --rl_epochs 100
```

### Training the Mixture of Experts Model

To train the Mixture of Experts model, run:

```bash
python train_moe_model.py --instruments Nifty "Bank Nifty" --timeframes 5 15 --num_experts 4 --k 2 --ensemble_size 3 --balance_instruments --balance_signals --use_enhanced_features
```

Both models use the three-stage RLHF pipeline:

1. **Behavioral Cloning**: Learns to mimic profitable trades from the signal column
2. **Reward Modeling**: Learns to predict the value of actions based on the signal column
3. **RL Fine-tuning**: Fine-tunes the model using PPO with KL constraint to the BC policy

## Evaluating the Models

### Evaluating the Decision Transformer Model

To evaluate the Decision Transformer model, run:

```bash
python evaluate_transformer.py --model_path models/trading_transformer.pt --instruments Nifty --timeframes 5 --plot
```

### Evaluating the Mixture of Experts Model

To evaluate the Mixture of Experts model, run:

```bash
python evaluate_moe_model.py --model_dir models --instruments Nifty --timeframes 5 --use_ensemble --ensemble_size 3 --plot --use_enhanced_features
```

Both evaluation scripts will generate performance plots and metrics for the specified instruments and timeframes.

## Live Trading

To run the model in live trading mode, run:

```bash
python live_trading.py
```

This will connect to the broker API, fetch market data, and make trading decisions based on the model's predictions.

## Configuration

The model and training parameters can be configured in `config.py`. The main configuration sections are:

- `TRANSFORMER_CONFIG`: Configuration for the transformer model
- `TRAINING_CONFIG`: Configuration for the training pipeline
- `RLHF_CONFIG`: Configuration for the RLHF components

## Signal Column

The signal column is used for RLHF during training but is never exposed to the model during inference. The signal values are:

- `0`: Hold
- `1`: Buy target hit (profitable long trade)
- `2`: Buy stop-loss hit (unprofitable long trade)
- `3`: Sell target hit (profitable short trade)
- `4`: Sell stop-loss hit (unprofitable short trade)

## Enhanced Features

The enhanced feature engineering includes:

1. **Multi-timeframe Features**: Captures different time horizons for better context
2. **Market Regime Detection**: Identifies trending, ranging, and volatile market conditions
3. **Pattern Recognition**: Detects candlestick patterns like inside bars, outside bars, and pin bars
4. **Relative Strength**: Measures price momentum and strength across different timeframes
5. **Volatility Metrics**: Normalized ATR, Bollinger Bands, and Keltner Channels
6. **Statistical Features**: Skewness, kurtosis, and autocorrelation of price movements
7. **Indian Market Specifics**: Days to expiry for Indian index options (last Thursday of month)
8. **Normalized Features**: Z-score normalization for better model training

## Model Architectures

### Decision Transformer Architecture

The Decision Transformer model architecture includes several enhancements:

1. **Transformer Encoder**: Processes the market data sequence
2. **Market Regime Detection**: Uses prototypical networks to detect market regimes
3. **Task Embedding**: Embeds instrument and timeframe information
4. **Position Features**: Incorporates position information into the model
5. **Risk Assessment**: Predicts the probability of stop-loss being hit

### Mixture of Experts (MoE) Architecture

The MoE model architecture extends the Decision Transformer with specialized expert networks:

1. **Expert Networks**: Multiple specialized networks that focus on different market regimes or patterns
2. **Gating Network**: Routes inputs to the appropriate experts based on input features
3. **Sparse Activation**: Only activates a subset of experts for each input to improve efficiency
4. **Load Balancing**: Ensures all experts are utilized equally during training
5. **Ensemble Prediction**: Combines predictions from multiple MoE models for more robust inference

## Anti-Overfitting Measures

To prevent overfitting and ensure good performance in live trading, the following measures are implemented:

1. **Ensemble Models**: Multiple models are trained and combined for more robust predictions
2. **Dropout and Regularization**: Prevents the model from memorizing the training data
3. **Cross-Validation**: Ensures the model generalizes well to unseen data
4. **Balanced Sampling**: Prevents bias towards specific instruments or signal types
5. **Asymmetric Loss Functions**: Handles the imbalance between profitable and unprofitable trades
6. **Uncertainty Estimation**: Measures prediction uncertainty and avoids high-uncertainty trades
7. **Curriculum Learning**: Starts with easier patterns and gradually increases difficulty
8. **Data Augmentation**: Creates synthetic training examples for better generalization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for the PPO implementation
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Gymnasium](https://gymnasium.farama.org/) for the reinforcement learning environment
- [pandas-ta](https://github.com/twopirllc/pandas-ta) for technical indicators
