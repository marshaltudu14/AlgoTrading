# Active Context: AlgoTrading System (2025-04-08 ~9:33 PM)

---

### **Current State**

- The project is modularized with clear separation of concerns.
- Fyers authentication is working and centralized.
- Data fetching functions retrieve recent historical candles successfully.
- A `DataProcessor` class:
  - Converts timestamps to IST.
  - Cleans data, drops duplicates and missing.
  - Adds EMA and ATR indicators using `pandas_ta`.
- The main script fetches and processes data **without errors**.
- Placeholder WebSocket subscriptions are initialized.
- The system is ready for strategy development and real-time integration.

---

### **Design Goals**

- Fully automated multi-instrument trading bot (options, futures, stocks).
- Multiple strategies: Inside Candle, EMA crossover, ML models.
- ATR-based risk management.
- Event-driven real-time trading using WebSockets.
- Configurable timeframes and instruments.
- Future ML integration with feature engineering and model inference.

---

### **Next Steps**

1. **Implement strategy modules:**
   - Inside Candle breakout.
   - EMA crossover.
2. **Integrate strategy signals into the main pipeline.**
3. **Develop order management and risk modules.**
4. **Replace placeholders with real WebSocket handlers.**
5. **Add a web-based UI (Streamlit or Dash) for control and visualization.**
6. **Prepare for ML model integration.**

---

### **Notes**

- The data pipeline is verified and operational.
- The architecture supports incremental, testable development.
- The memory bank will be updated continuously as progress is made.
