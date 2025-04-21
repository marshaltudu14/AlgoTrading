# AlgoBot Plan and Roadmap

## 1. Overview
- Purpose: A pure RL‑based autonomous trading bot.
- Scope: Single instrument & timeframe in live trading, multi‑task training across indices & intervals.

## 2. Environment Design
- Use **Gymnasium** (replacement for Gym).
- Custom Gymnasium env wrapper around historical data and live websocket.
- State: recent price windows, technical indicators, position status.
- Action: discrete or continuous (e.g. buy/hold/sell, or order size).
0
## 3. Reward Structure
- ΔPnL − λ·TransactionCost − μ·DrawdownPenalty
- Hyperparams λ, μ to balance risk & turnover.

## 4. RL Algorithm Options
- **Stable‑Baselines3** (PyTorch) with Gymnasium support: PPO, SAC, TQC
- **Ray RLlib** for scalable multi‑task
- **Custom** PyTorch/TensorFlow models if bespoke architectures needed

## 5. Meta‑RL Approach
- Tasks = (instrument, timeframe) combos.
- Meta‑train with **Learn2Learn (MAML/RL²)** on SB3 policies.
- Fast adaptation (few‑shot fine‑tuning) in live market.
- **Chunked Multi‑task Training**: Split tasks into configurable chunks (via `RL_CHUNK_SIZE`, default 8), train the RL² model iteratively per chunk with intermediate checkpoints (`models/rl2_multitask_chunk_{n}`) and memory cleanup to handle large datasets without overloading RAM.

## 6. Implementation Stack
- Python 3.12, Windows (use `py` for commands)
- Gymnasium
- stable‑baselines3 or RLlib
- learn2learn
- PyTorch (core)
- Pandas, NumPy

## 7. Deployment Strategy
1. Train meta‑agent on historical_data/ folder.
2. Fine‑tune on live websocket data for target instrument/timeframe.
3. Gate orders with risk checks.
4. Logging & monitoring.

## 8. Future Plans
- Add position sizing via risk parity.
- Incorporate on‑chain or alternative data.
- Multi‑asset extension.
- UI dashboard (Streamlit/React) for oversight.

## 9. Data Preprocessing & Labeling
- Use **pandas_ta** to compute ATR (14‑period).
- For each row, calculate stop‑loss and target with risk–reward 1:2:
  - SL_buy = close − ATR; Target_buy = close + 2×ATR
  - SL_sell = close + ATR; Target_sell = close − 2×ATR
- Look ahead row‑wise (e.g. with `tqdm`) until SL or target hit, then label:
  - 0: HOLD
  - 1: BUY_TARGET_HIT
  - 2: BUY_SL_HIT
  - 3: SELL_TARGET_HIT
  - 4: SELL_SL_HIT
- Save processed CSVs to `processed_data/`.
- Before overwriting, compare new DataFrame columns with existing file and skip if identical.

## 10. RLHF & Advanced Reward Modeling
- **Hybrid Pipeline**: Behavior Cloning → Preference Elicitation → Reward Modeling → RL Fine‑tuning
- **Demonstrations**: Pretrain policy network on labeled “guaranteed profit” entries via BC.
- **Preference Data**: Sample trajectory clips; collect pairwise human preferences (or heuristics) on “better” trades.
- **Reward Model**: Train a neural network R(·) on (state,action) to predict scaled human score.
- **Combined Reward** at each step:
  ```
  rₜ = w₁·r_PnLₜ  + w₂·r_RiskControlₜ  + w₃·R(stateₜ,actionₜ)
       + w₄·r_Intrinsicₜ  – w₅·Penaltyₜ
  ```
  - **r_PnL**: normalized profit from ΔPrice·lot_size minus cost
  - **r_RiskControl**: CVaR or max‑drawdown penalty
  - **R(·)**: learned human‑preference reward
  - **r_Intrinsic**: curiosity bonus via prediction error of world model
  - **Penalty**: transaction cost, theta decay, turnover
- **Meta‑Optimization**: Use gradient‑based meta‑RL (MAML) to adapt weights wᵢ across tasks (instruments/TFs).
- **Exploration**: Leverage adversarial domain randomization on volatility and liquidity.

---
*Generated on 2025‑04‑20*
