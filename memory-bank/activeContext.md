# Active Context: AlgoTrading System (2025-04-08 ~8:02 PM)

---

### **Current State**

- The project has been reset and modularized with a clean architecture.
- Fyers authentication is working and modularized.
- Data fetching functions for both real-time and training data have been implemented in `data/fetcher.py`.
- The system is designed to support both traditional rule-based strategies and future ML-driven strategies.

---

### **Design Goals**

- **Fully automated multi-instrument trading bot** (options, futures, stocks).
- **Multiple strategies**: Inside Candle, EMA crossover, and ML-based models.
- **ATR-based risk management** for SL/TP.
- **Modular architecture** with clear separation:
  - Authentication
  - Data fetching and processing
  - Strategy modules
  - Risk management
  - Order and position management
  - Real-time data and execution
  - Utilities
- **Machine Learning integration**:
  - Fetch and process historical data for training.
  - Feature engineering pipeline.
  - Model inference for entry/exit signals.
  - Real-time data feeding into models.

---

### **Next Steps**

1. **Feature Engineering**:
   - Implement in `data/processor.py`.
   - Calculate features like EMA, ATR, candle patterns, etc.
2. **Strategy Modules**:
   - Implement Inside Candle and EMA crossover in `strategies/`.
   - Design ML strategy interface for future models.
3. **Model Integration**:
   - Train models in Jupyter notebooks.
   - Export trained models.
   - Load and run inference in production bot.
4. **Real-Time Pipeline**:
   - Use WebSocket for tick data and order updates.
   - Feed real-time data into strategies and models.
   - Execute trades automatically.
5. **Testing and Deployment**:
   - Test each module independently.
   - Integrate and test full pipeline in paper trading.
   - Deploy live with monitoring.

---

### **Notes**

- Use **Jupyter notebooks** for ML development, visualization, and experimentation.
- Use **terminal scripts** for live trading and automation.
- Continuously update the memory bank as the project evolves.
