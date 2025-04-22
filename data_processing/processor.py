import os
import pandas as pd
import numpy as np
# Monkey-patch for pandas_ta compatibility (numpy.NaN alias)
setattr(np, 'NaN', np.nan)
import pandas_ta as ta
from tqdm import tqdm

def process_df(df, rr_ratio=2):
    df = df.copy()
    # Compute ATR (14-period)
    df['ATR'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
    n = len(df)
    signals = np.zeros(n, dtype=int)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    atr = df['ATR'].values

    for i in tqdm(range(n-1), desc='Processing signals'):
        if pd.isnull(atr[i]):
            continue
        entry = close[i]
        atr_i = atr[i]
        # Stop-loss and target for buy
        sl_buy = entry - atr_i
        tgt_buy = entry + atr_i * rr_ratio
        # Stop-loss and target for sell
        sl_sell = entry + atr_i
        tgt_sell = entry - atr_i * rr_ratio
        label = 0
        for j in range(i+1, n):
            if high[j] >= tgt_buy:
                label = 1
                break
            elif low[j] <= sl_buy:
                label = 2
                break
            elif low[j] <= tgt_sell:
                label = 3
                break
            elif high[j] >= sl_sell:
                label = 4
                break
        signals[i] = label

    df['signal'] = signals
    # Drop initial rows with NaN from ATR or lookback
    df.dropna(axis=0, how='any', inplace=True)
    return df


def process_all(input_dir='historical_data', output_dir='processed_data', rr_ratio=2):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.csv'):
            continue
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        # Determine expected columns from raw header
        raw_cols = list(pd.read_csv(in_path, nrows=0).columns)
        expected = set(raw_cols) | {'ATR', 'signal'}
        # Skip full processing if processed file already has desired columns
        if os.path.exists(out_path):
            existing = set(pd.read_csv(out_path, nrows=0).columns)
            if existing == expected:
                print(f"No change in features for {fname}, skipping processing.")
                continue
        # Heavy processing
        df = pd.read_csv(in_path)
        processed = process_df(df, rr_ratio=rr_ratio)
        processed.to_csv(out_path, index=False)
        print(f"Saved processed file: {out_path}")

if __name__ == '__main__':
    process_all()
