import os
import pandas as pd
import numpy as np
# Monkey-patch for pandas_ta compatibility (numpy.NaN alias)
setattr(np, 'NaN', np.nan)
import pandas_ta as ta
from tqdm import tqdm

def process_df(df, rr_ratio=2):
    df = df.copy()
    # === Price-based Features ===
    df['delta_close'] = df['close'].diff()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # === Trend Features ===
    windows = [5, 10, 20, 50, 100]
    for w in windows:
        df[f'SMA_{w}'] = ta.sma(close=df['close'], length=w)
        df[f'EMA_{w}'] = ta.ema(close=df['close'], length=w)
        df[f'WMA_{w}'] = ta.wma(close=df['close'], length=w)
        df[f'ROLL_STD_{w}'] = df['close'].rolling(w).std()

    # === Momentum Features ===
    df['RSI_14'] = ta.rsi(close=df['close'], length=14)
    df['MOM_10'] = ta.mom(close=df['close'], length=10)
    df['ROC_10'] = ta.roc(close=df['close'], length=10)
    # True Strength Index (multi-column)
    tsi = ta.tsi(close=df['close'], r=25, s=13)
    df = pd.concat([df, tsi], axis=1)

    # === Volatility/Channel Features ===
    df['ATR'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
    keltner = ta.kc(high=df['high'], low=df['low'], close=df['close'], length=20, scalar=1.5)
    df = pd.concat([df, keltner], axis=1)
    donchian = ta.donchian(high=df['high'], low=df['low'], length=20)
    df = pd.concat([df, donchian], axis=1)
    df['ROLL_VOL_14'] = np.log(df['close'] / df['close'].shift(1)).rolling(14).std()

    # === Volume-based Features (skipped if no valid volume) ===
    if 'volume' in df.columns and not (df['volume'].isnull().all() or (df['volume'] == 0).all()):
        df['OBV'] = ta.obv(close=df['close'], volume=df['volume'])
        if 'datetime' in df.columns:
            dt = pd.to_datetime(df['datetime'])
            is_dup = dt.duplicated(keep='first')
            if is_dup.any():
                print('Dropping duplicate datetimes for VWAP calculation.')
            df_vwap = df.loc[~is_dup].set_index(dt[~is_dup])
            if df_vwap.index.is_unique:
                df['VWAP'] = ta.vwap(high=df_vwap['high'], low=df_vwap['low'], close=df_vwap['close'], volume=df_vwap['volume']).reindex(df.index)
            else:
                print('VWAP not computed: index not unique even after dropping duplicates.')
        else:
            print('VWAP not computed: no datetime column for index.')
        df['MFI'] = ta.mfi(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], length=14)
    else:
        print('No valid volume column: skipping OBV, VWAP, MFI.')

    # === Cycle & Oscillator Features ===
    macd = ta.macd(close=df['close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    df['CCI_20'] = ta.cci(high=df['high'], low=df['low'], close=df['close'], length=20)
    stochrsi = ta.stochrsi(close=df['close'], length=14, rsi_length=14, k=3, d=3)
    df = pd.concat([df, stochrsi], axis=1)

    # === Statistical Features ===
    for w in windows:
        df[f'SKEW_{w}'] = df['close'].rolling(w).skew()
        df[f'KURT_{w}'] = df['close'].rolling(w).kurt()
    df['ACF_1'] = df['close'].rolling(20).apply(lambda x: x.autocorr(lag=1), raw=False)

    # === Time Features ===
    if 'datetime' in df.columns:
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['datetime']).dt.weekday
        df['month'] = pd.to_datetime(df['datetime']).dt.month

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
        # Always process and dynamically save feature columns
        # Heavy processing
        df = pd.read_csv(in_path)
        processed = process_df(df, rr_ratio=rr_ratio)
        processed.to_csv(out_path, index=False)
        # Save dynamic feature columns for RL env (excluding base cols)
        exclude = {'signal', 'datetime', 'Unnamed: 0'}
        feature_cols = [c for c in processed.columns if c not in exclude]
        import json
        with open(out_path.replace('.csv', '.features.json'), 'w') as f:
            json.dump(feature_cols, f, indent=2)
        print(f"Saved processed file: {out_path} and features: {out_path.replace('.csv', '.features.json')}")

if __name__ == '__main__':
    process_all()
