# Strategy Documentation

This file documents the trading strategies implemented and tested within this project.

## 1. Inside Candle Breakout (Custom Backtester)

*   **File:** `src/custom_backtester.py` (Signal logic: `get_inside_candle_signals`, Execution: `run_custom_backtest`)
*   **Entry Logic:**
    *   Identifies an "inside bar" (previous bar's high/low is within the pre-previous bar's high/low).
    *   Enters LONG on the open of the bar *after* the inside bar if that bar's high breaks *above* the high of the "mother" candle (the pre-previous bar).
    *   Enters SHORT on the open of the bar *after* the inside bar if that bar's low breaks *below* the low of the "mother" candle.
*   **Exit Logic:**
    *   Uses Stop Loss (SL) and Take Profit (TP) levels calculated at the time of entry.
    *   SL = Entry Price +/- (ATR * SL Multiplier)
    *   TP = Entry Price +/- (SL Distance * RR Ratio)
    *   ATR is taken from the *signal bar* (the inside bar itself).
*   **Parameters (Defaults in `run_custom_backtest.py`):**
    *   `stop_loss_atr_multiplier`: 1.5
    *   `risk_reward_ratio`: 2.0
    *   `atr_period`: 14
    *   `commission_per_trade`: 5.0 (fixed currency amount per leg)
*   **Backtest Results (Nifty 5min, 2022-03-07 to 2024-11-27):**
    *   Total Return: ~1769%
    *   Win Rate: ~46.5%
    *   Profit Factor: ~1.68
    *   Max Drawdown: ~13.4%
    *   Number of Trades: 2090
*   **Notes:**
    *   Relies on pre-calculated ATR (`atr_14`) in the input data.
    *   Uses a manual backtesting loop, offering fine control but requiring careful implementation.
    *   High trade frequency.


================================================================================
Batch Run: 2025-04-06 12:54:26
Strategy: Inside Candle Breakout (Custom)
Parameters: SL ATR Mult=1.5, RR=2.0, ATR Period=14, Commission=10.00 (Round Trip)
--------------------------------------------------------------------------------



================================================================================
Batch Run: 2025-04-06 13:00:04
Strategy: Inside Candle Breakout (Custom)
Parameters: SL ATR Mult=1.5, RR=2.0, ATR Period=14, Commission=10.00 (Round Trip)
--------------------------------------------------------------------------------

| Instrument   |   Timeframe (min) |   Lot Size | Initial Cash   | Final Equity   |   Total Return [%] |   Win Rate [%] |   Number of Trades |   Number of Wins |   Number of Losses |   Profit Factor |   Max Drawdown [%] |
|:-------------|------------------:|-----------:|:---------------|:---------------|-------------------:|---------------:|-------------------:|-----------------:|-------------------:|----------------:|-------------------:|
| Bank_Nifty   |               240 |         30 | 1.00 L         | 5.97 L         |             496.92 |          44.16 |                 77 |               34 |                 43 |            1.66 |              42.33 |
| Bank_Nifty   |               180 |         30 | 1.00 L         | 7.95 L         |             694.81 |          44.05 |                168 |               74 |                 94 |            1.57 |              35.98 |
| Bank_Nifty   |               120 |         30 | 1.00 L         | 12.69 L        |            1168.73 |          48.43 |                223 |              108 |                115 |            1.86 |              27.88 |
| Bank_Nifty   |                60 |         30 | 1.00 L         | 7.76 L         |             676.09 |          42.27 |                291 |              123 |                168 |            1.44 |             145.12 |
| Bank_Nifty   |                45 |         30 | 1.00 L         | 9.91 L         |             890.9  |          44.64 |                345 |              154 |                191 |            1.57 |             113.72 |
| Bank_Nifty   |                30 |         30 | 1.00 L         | 12.66 L        |            1166.19 |          46.32 |                462 |              214 |                248 |            1.68 |              71.25 |
| Bank_Nifty   |                20 |         30 | 1.00 L         | 16.37 L        |            1536.8  |          47.91 |                622 |              298 |                324 |            1.85 |              15.48 |
| Bank_Nifty   |                15 |         30 | 1.00 L         | 14.50 L        |            1349.98 |          45.17 |                786 |              355 |                431 |            1.62 |              62.84 |
| Bank_Nifty   |                10 |         30 | 1.00 L         | 15.12 L        |            1411.72 |          43.9  |               1180 |              518 |                662 |            1.53 |              24.07 |
| Bank_Nifty   |                 5 |         30 | 1.00 L         | 19.61 L        |            1861.23 |          44.98 |               2021 |              909 |               1112 |            1.58 |              19.86 |
| Bank_Nifty   |                 3 |         30 | 1.00 L         | 26.10 L        |            2510.06 |          45.17 |               3376 |             1525 |               1851 |            1.62 |              10.46 |
| Bank_Nifty   |                 2 |         30 | 1.00 L         | 32.53 L        |            3153.43 |          44.66 |               5213 |             2328 |               2885 |            1.63 |              26.39 |
| Bankex       |               240 |         15 | 1.00 L         | 4.49 L         |             349.32 |          47.14 |                 70 |               33 |                 37 |            1.98 |              19.9  |
| Bankex       |               180 |         15 | 1.00 L         | 2.89 L         |             188.89 |          38.89 |                180 |               70 |                110 |            1.23 |              58.89 |
| Bankex       |               120 |         15 | 1.00 L         | 4.11 L         |             311.35 |          41.79 |                201 |               84 |                117 |            1.39 |              59.91 |
| Bankex       |                60 |         15 | 1.00 L         | 3.96 L         |             296.41 |          40.07 |                297 |              119 |                178 |            1.32 |              57.42 |
| Bankex       |                45 |         15 | 1.00 L         | 4.69 L         |             368.55 |          41.48 |                352 |              146 |                206 |            1.39 |              75.02 |
| Bankex       |                30 |         15 | 1.00 L         | 5.50 L         |             450.47 |          42.92 |                431 |              185 |                246 |            1.48 |              24.53 |
| Bankex       |                20 |         15 | 1.00 L         | 8.39 L         |             738.62 |          45.64 |                631 |              288 |                343 |            1.7  |              16.08 |
| Bankex       |                15 |         15 | 1.00 L         | 7.67 L         |             666.64 |          43.9  |                770 |              338 |                432 |            1.55 |              16.69 |
| Bankex       |                10 |         15 | 1.00 L         | 9.92 L         |             891.9  |          45.33 |               1145 |              519 |                626 |            1.64 |              12.39 |
| Bankex       |                 5 |         15 | 1.00 L         | 10.50 L        |             949.87 |          43.98 |               2069 |              910 |               1159 |            1.53 |              13.37 |
| Bankex       |                 3 |         15 | 1.00 L         | 13.90 L        |            1290.37 |          44.16 |               3410 |             1506 |               1904 |            1.59 |              11.27 |
| Bankex       |                 2 |         15 | 1.00 L         | 17.58 L        |            1657.77 |          44.66 |               5314 |             2373 |               2941 |            1.62 |               7.74 |
| Finnifty     |               240 |         40 | 1.00 L         | 3.07 L         |             206.7  |          40    |                 80 |               32 |                 48 |            1.43 |              46.32 |
| Finnifty     |               180 |         40 | 1.00 L         | 3.16 L         |             215.99 |          38.38 |                185 |               71 |                114 |            1.26 |              42.83 |
| Finnifty     |               120 |         40 | 1.00 L         | 4.31 L         |             331.45 |          41.23 |                211 |               87 |                124 |            1.41 |              32.33 |
| Finnifty     |                60 |         40 | 1.00 L         | 5.79 L         |             478.98 |          42.71 |                295 |              126 |                169 |            1.57 |              41.16 |
| Finnifty     |                45 |         40 | 1.00 L         | 4.72 L         |             372.08 |          39.94 |                343 |              137 |                206 |            1.4  |              32    |
| Finnifty     |                30 |         40 | 1.00 L         | 7.25 L         |             625.09 |          44.52 |                429 |              191 |                238 |            1.7  |              41.3  |
| Finnifty     |                20 |         40 | 1.00 L         | 8.07 L         |             706.58 |          46.42 |                614 |              285 |                329 |            1.66 |              15.65 |
| Finnifty     |                15 |         40 | 1.00 L         | 7.99 L         |             698.59 |          44.41 |                752 |              334 |                418 |            1.59 |              13.77 |
| Finnifty     |                10 |         40 | 1.00 L         | 10.25 L        |             925.24 |          45.37 |               1166 |              529 |                637 |            1.64 |              11.75 |
| Finnifty     |                 5 |         40 | 1.00 L         | 10.91 L        |             990.87 |          44.21 |               1968 |              870 |               1098 |            1.57 |               8.83 |
| Finnifty     |                 3 |         40 | 1.00 L         | 15.15 L        |            1415.5  |          45.72 |               3314 |             1515 |               1799 |            1.64 |               8.57 |
| Finnifty     |                 2 |         40 | 1.00 L         | 18.32 L        |            1732.3  |          45.04 |               5146 |             2318 |               2828 |            1.64 |               8.19 |
| Nifty        |               240 |         75 | 1.00 L         | 6.35 L         |             534.89 |          47.83 |                 69 |               33 |                 36 |            1.98 |              31.33 |
| Nifty        |               180 |         75 | 1.00 L         | 6.65 L         |             565.35 |          42.05 |                176 |               74 |                102 |            1.51 |              41.88 |
| Nifty        |               120 |         75 | 1.00 L         | 7.30 L         |             629.76 |          43.33 |                210 |               91 |                119 |            1.54 |              36.44 |
| Nifty        |                60 |         75 | 1.00 L         | 7.54 L         |             653.72 |          42.35 |                307 |              130 |                177 |            1.48 |              67.94 |
| Nifty        |                45 |         75 | 1.00 L         | 9.50 L         |             850.38 |          45.18 |                363 |              164 |                199 |            1.63 |              53.34 |
| Nifty        |                30 |         75 | 1.00 L         | 10.53 L        |             953.46 |          44.23 |                468 |              207 |                261 |            1.65 |              30.11 |
| Nifty        |                20 |         75 | 1.00 L         | 11.85 L        |            1084.56 |          46.71 |                653 |              305 |                348 |            1.65 |              14.53 |
| Nifty        |                15 |         75 | 1.00 L         | 12.75 L        |            1175.31 |          46.11 |                796 |              367 |                429 |            1.66 |              21.1  |
| Nifty        |                10 |         75 | 1.00 L         | 15.47 L        |            1447.46 |          46.11 |               1143 |              527 |                616 |            1.73 |              12.53 |
| Nifty        |                 5 |         75 | 1.00 L         | 18.69 L        |            1768.57 |          46.51 |               2090 |              972 |               1118 |            1.68 |              13.37 |
| Nifty        |                 3 |         75 | 1.00 L         | 22.48 L        |            2147.57 |          45.58 |               3447 |             1571 |               1876 |            1.65 |              17    |
| Nifty        |                 2 |         75 | 1.00 L         | 26.54 L        |            2553.79 |          44.7  |               5452 |             2437 |               3015 |            1.62 |               6.26 |
| Sensex       |               240 |         20 | 1.00 L         | 5.63 L         |             463.13 |          46.97 |                 66 |               31 |                 35 |            2.02 |              45.31 |
| Sensex       |               180 |         20 | 1.00 L         | 6.97 L         |             596.72 |          44.05 |                168 |               74 |                 94 |            1.65 |              39.57 |
| Sensex       |               120 |         20 | 1.00 L         | 9.02 L         |             802.44 |          48    |                200 |               96 |                104 |            1.87 |              38.89 |
| Sensex       |                60 |         20 | 1.00 L         | 5.95 L         |             494.81 |          40.98 |                305 |              125 |                180 |            1.4  |              34.1  |
| Sensex       |                45 |         20 | 1.00 L         | 8.22 L         |             722.29 |          42.94 |                354 |              152 |                202 |            1.61 |              30.92 |
| Sensex       |                30 |         20 | 1.00 L         | 9.61 L         |             860.75 |          45.02 |                462 |              208 |                254 |            1.68 |              26.03 |
| Sensex       |                20 |         20 | 1.00 L         | 9.34 L         |             833.78 |          44.8  |                654 |              293 |                361 |            1.55 |              34.78 |
| Sensex       |                15 |         20 | 1.00 L         | 11.50 L        |            1050.36 |          45.06 |                790 |              356 |                434 |            1.67 |              13.21 |
| Sensex       |                10 |         20 | 1.00 L         | 14.64 L        |            1363.61 |          47.57 |               1133 |              539 |                594 |            1.79 |              14.63 |
| Sensex       |                 5 |         20 | 1.00 L         | 16.34 L        |            1533.71 |          46.1  |               2065 |              952 |               1113 |            1.67 |              18.7  |
| Sensex       |                 3 |         20 | 1.00 L         | 18.94 L        |            1793.83 |          45.34 |               3485 |             1580 |               1905 |            1.61 |              14.71 |
| Sensex       |                 2 |         20 | 1.00 L         | 22.94 L        |            2193.88 |          45.02 |               5324 |             2397 |               2927 |            1.63 |              16.61 |

================================================================================
