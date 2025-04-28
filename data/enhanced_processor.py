"""
Enhanced data processor for AlgoTrading.
Includes multi-timeframe features, market regime detection, and pattern recognition.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
# Monkey-patch for pandas_ta compatibility (numpy.NaN alias)
setattr(np, 'NaN', np.nan)
import pandas_ta as ta
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler

from core.logging_setup import get_logger
from core.constants import MarketRegime

logger = get_logger(__name__)


def process_df(df, rr_ratio=2, multi_timeframe=True, normalize=True):
    """
    Process DataFrame with enhanced feature engineering.
    
    Args:
        df: Input DataFrame with OHLC data
        rr_ratio: Risk-reward ratio for signal generation
        multi_timeframe: Whether to include multi-timeframe features
        normalize: Whether to normalize features
        
    Returns:
        Processed DataFrame with enhanced features
    """
    logger.info("Processing DataFrame with enhanced features")
    df = df.copy()
    
    # Ensure datetime is in the right format
    if 'datetime' in df.columns:
        if isinstance(df['datetime'].iloc[0], (int, float)):
            df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        else:
            df['datetime'] = pd.to_datetime(df['datetime'])
    
    # === Price-based Features ===
    df['delta_close'] = df['close'].diff()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['return_volatility'] = df['log_return'].rolling(20).std()
    
    # Normalized price channels
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # === Multi-timeframe Features ===
    windows = [5, 10, 20, 50, 100]
    
    # === Trend Features ===
    for w in windows:
        # Moving averages
        df[f'SMA_{w}'] = ta.sma(close=df['close'], length=w)
        df[f'EMA_{w}'] = ta.ema(close=df['close'], length=w)
        df[f'WMA_{w}'] = ta.wma(close=df['close'], length=w)
        df[f'ROLL_STD_{w}'] = df['close'].rolling(w).std()
        
        # Normalized price relative to moving averages
        df[f'CLOSE_REL_SMA_{w}'] = df['close'] / df[f'SMA_{w}'] - 1
        df[f'CLOSE_REL_EMA_{w}'] = df['close'] / df[f'EMA_{w}'] - 1
        
        # Crossover signals
        if w < 50:  # Only for shorter windows
            df[f'EMA_CROSS_{w}'] = np.where(
                df[f'EMA_{w}'] > df[f'SMA_{w}'], 
                1, 
                np.where(df[f'EMA_{w}'] < df[f'SMA_{w}'], -1, 0)
            )
    
    # === Momentum Features ===
    # RSI with multiple lookbacks
    for w in [7, 14, 21]:
        df[f'RSI_{w}'] = ta.rsi(close=df['close'], length=w)
    
    # RSI divergence
    df['RSI_DIFF'] = df['RSI_14'] - df['RSI_14'].shift(5)
    
    # Momentum indicators
    df['MOM_10'] = ta.mom(close=df['close'], length=10)
    df['ROC_10'] = ta.roc(close=df['close'], length=10)
    
    # True Strength Index
    tsi = ta.tsi(close=df['close'], r=25, s=13)
    df = pd.concat([df, tsi], axis=1)
    
    # === Volatility/Channel Features ===
    # ATR with multiple lookbacks
    for w in [7, 14, 21]:
        df[f'ATR_{w}'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=w)
    
    # Use ATR_14 as the default ATR
    df['ATR'] = df['ATR_14']
    
    # Normalized ATR (volatility relative to price)
    df['NATR'] = df['ATR'] / df['close'] * 100
    
    # Bollinger Bands
    bbands = ta.bbands(close=df['close'], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    # Normalized Bollinger Band position
    df['BB_POSITION'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
    
    # Keltner Channels
    keltner = ta.kc(high=df['high'], low=df['low'], close=df['close'], length=20, scalar=1.5)
    df = pd.concat([df, keltner], axis=1)
    
    # Donchian Channels
    donchian = ta.donchian(high=df['high'], low=df['low'], length=20)
    df = pd.concat([df, donchian], axis=1)
    
    # Rolling volatility
    df['ROLL_VOL_14'] = np.log(df['close'] / df['close'].shift(1)).rolling(14).std()
    
    # === Volume Features (if available) ===
    if 'volume' in df.columns and not (df['volume'].isnull().all() or (df['volume'] == 0).all()):
        # On-Balance Volume
        df['OBV'] = ta.obv(close=df['close'], volume=df['volume'])
        
        # Volume-Weighted Average Price
        if 'datetime' in df.columns:
            dt = pd.to_datetime(df['datetime'])
            is_dup = dt.duplicated(keep='first')
            if is_dup.any():
                logger.warning('Dropping duplicate datetimes for VWAP calculation.')
            df_vwap = df.loc[~is_dup].set_index(dt[~is_dup])
            if df_vwap.index.is_unique:
                df['VWAP'] = ta.vwap(high=df_vwap['high'], low=df_vwap['low'], 
                                     close=df_vwap['close'], volume=df_vwap['volume']).reindex(df.index)
            else:
                logger.warning('VWAP not computed: index not unique even after dropping duplicates.')
        else:
            logger.warning('VWAP not computed: no datetime column for index.')
        
        # Money Flow Index
        df['MFI'] = ta.mfi(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], length=14)
        
        # Volume relative to moving average
        df['VOL_REL_MA'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Price-volume correlation
        df['PRICE_VOL_CORR'] = df['close'].rolling(20).corr(df['volume'])
    else:
        logger.warning('No valid volume column: skipping OBV, VWAP, MFI.')
    
    # === Cycle & Oscillator Features ===
    # MACD
    macd = ta.macd(close=df['close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    
    # MACD histogram normalized
    if 'MACDh_12_26_9' in df.columns:
        df['MACD_NORM'] = df['MACDh_12_26_9'] / df['close'] * 100
    
    # Commodity Channel Index
    df['CCI_20'] = ta.cci(high=df['high'], low=df['low'], close=df['close'], length=20)
    
    # Stochastic RSI
    stochrsi = ta.stochrsi(close=df['close'], length=14, rsi_length=14, k=3, d=3)
    df = pd.concat([df, stochrsi], axis=1)
    
    # Awesome Oscillator
    ao = ta.ao(high=df['high'], low=df['low'])
    df = pd.concat([df, ao], axis=1)
    
    # === Pattern Recognition ===
    # Inside bar pattern
    df['INSIDE_BAR'] = ((df['high'] <= df['high'].shift(1)) & 
                         (df['low'] >= df['low'].shift(1))).astype(int)
    
    # Outside bar pattern
    df['OUTSIDE_BAR'] = ((df['high'] > df['high'].shift(1)) & 
                          (df['low'] < df['low'].shift(1))).astype(int)
    
    # Pin bar pattern (approximation)
    upper_wick = df['high'] - np.maximum(df['open'], df['close'])
    lower_wick = np.minimum(df['open'], df['close']) - df['low']
    body = abs(df['close'] - df['open'])
    df['PIN_BAR_UP'] = ((lower_wick > 2 * body) & (upper_wick < 0.3 * lower_wick)).astype(int)
    df['PIN_BAR_DOWN'] = ((upper_wick > 2 * body) & (lower_wick < 0.3 * upper_wick)).astype(int)
    
    # === Statistical Features ===
    for w in windows:
        df[f'SKEW_{w}'] = df['close'].rolling(w).skew()
        df[f'KURT_{w}'] = df['close'].rolling(w).kurt()
    
    # Autocorrelation
    df['ACF_1'] = df['close'].rolling(20).apply(lambda x: x.autocorr(lag=1), raw=False)
    df['ACF_5'] = df['close'].rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False)
    
    # Z-score of returns
    df['RETURN_ZSCORE'] = stats.zscore(df['log_return'].fillna(0), nan_policy='omit')
    
    # === Market Regime Features ===
    # Trend strength indicator
    adx = ta.adx(high=df['high'], low=df['low'], close=df['close'])
    if 'ADX_14' in adx.columns:
        df['ADX'] = adx['ADX_14']
    
    # Volatility regime
    vol_ma = df['ROLL_VOL_14'].rolling(30).mean()
    df['VOL_REGIME'] = np.where(
        df['ROLL_VOL_14'] > vol_ma * 1.2, 
        MarketRegime.VOLATILE.value,  # High volatility
        np.where(
            df['ROLL_VOL_14'] < vol_ma * 0.8, 
            MarketRegime.RANGING.value,  # Low volatility
            MarketRegime.TRENDING.value   # Normal volatility
        )
    )
    
    # Trend regime based on ADX
    if 'ADX' in df.columns:
        df['TREND_REGIME'] = np.where(
            df['ADX'] > 25, 
            1,  # Trending
            0   # Ranging
        )
    
    # === Time Features ===
    if 'datetime' in df.columns:
        dt = pd.to_datetime(df['datetime'])
        df['hour'] = dt.dt.hour
        df['minute'] = dt.dt.minute
        df['day_of_week'] = dt.dt.dayofweek
        df['day_of_month'] = dt.dt.day
        df['month'] = dt.dt.month
        df['quarter'] = dt.dt.quarter
        
        # Market session (India)
        df['SESSION'] = np.where(
            (dt.dt.hour >= 9) & (dt.dt.hour < 12),
            0,  # Morning session
            np.where(
                (dt.dt.hour >= 12) & (dt.dt.hour < 15),
                1,  # Afternoon session
                2   # Evening session
            )
        )
        
        # Days to expiry (for Indian markets - last Thursday of month)
        def days_to_expiry(date):
            # Find the last Thursday of the current month
            year, month = date.year, date.month
            # Get the last day of the month
            if month == 12:
                next_month = datetime(year + 1, 1, 1)
            else:
                next_month = datetime(year, month + 1, 1)
            last_day = (next_month - pd.Timedelta(days=1)).day
            
            # Find the last Thursday
            last_thursday = last_day
            while datetime(year, month, last_thursday).weekday() != 3:  # 3 is Thursday
                last_thursday -= 1
            
            # Calculate days to expiry
            expiry_date = datetime(year, month, last_thursday)
            days = (expiry_date - date).days
            
            # If already passed this month's expiry, look at next month
            if days < 0:
                if month == 12:
                    next_expiry = days_to_expiry(datetime(year + 1, 1, 1))
                else:
                    next_expiry = days_to_expiry(datetime(year, month + 1, 1))
                return next_expiry
            
            return days
        
        try:
            df['DAYS_TO_EXPIRY'] = dt.apply(days_to_expiry)
        except Exception as e:
            logger.warning(f"Could not calculate days to expiry: {e}")
    
    # === Signal Generation ===
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
        for j in range(i+1, min(i+50, n)):  # Limit lookahead to 50 bars
            if high[j] >= tgt_buy:
                label = 1  # Buy target hit
                break
            elif low[j] <= sl_buy:
                label = 2  # Buy SL hit
                break
            elif low[j] <= tgt_sell:
                label = 3  # Sell target hit
                break
            elif high[j] >= sl_sell:
                label = 4  # Sell SL hit
                break
        signals[i] = label
    
    df['signal'] = signals
    
    # === Feature Normalization ===
    if normalize:
        logger.info("Normalizing features")
        # Identify numeric columns for normalization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude certain columns from normalization
        exclude_cols = ['datetime', 'signal', 'hour', 'minute', 'day_of_week', 
                         'day_of_month', 'month', 'quarter', 'SESSION', 'DAYS_TO_EXPIRY',
                         'VOL_REGIME', 'TREND_REGIME', 'INSIDE_BAR', 'OUTSIDE_BAR',
                         'PIN_BAR_UP', 'PIN_BAR_DOWN']
        
        # Filter columns to normalize
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
        
        # Apply normalization
        for col in cols_to_normalize:
            # Skip columns with all NaN
            if df[col].isna().all():
                continue
                
            # Get mean and std, ignoring NaN
            mean = df[col].mean()
            std = df[col].std()
            
            # Skip if std is 0 or NaN
            if std == 0 or pd.isna(std):
                continue
                
            # Normalize to z-scores
            df[col] = (df[col] - mean) / std
    
    # Drop initial rows with NaN from ATR or lookback
    df.dropna(axis=0, how='any', inplace=True)
    
    logger.info(f"Processed DataFrame with {len(df)} rows and {len(df.columns)} columns")
    return df


def process_all(input_dir='historical_data', output_dir='processed_data', rr_ratio=2, normalize=True):
    """
    Process all CSV files in the input directory and save to output directory.
    
    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Directory to save processed CSV files
        rr_ratio: Risk-reward ratio for signal generation
        normalize: Whether to normalize features
    """
    logger.info(f"Processing all files in {input_dir} with enhanced features")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    for fname in csv_files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, f"enhanced_{fname}")
        
        logger.info(f"Processing {fname} with enhanced features")
        
        # Load data
        df = pd.read_csv(in_path)
        
        # Process data
        processed = process_df(df, rr_ratio=rr_ratio, normalize=normalize)
        
        # Save processed data
        processed.to_csv(out_path, index=False)
        
        # Save feature columns for RL env
        exclude = {'signal', 'datetime', 'Unnamed: 0'}
        feature_cols = [c for c in processed.columns if c not in exclude]
        
        import json
        with open(out_path.replace('.csv', '.features.json'), 'w') as f:
            json.dump(feature_cols, f, indent=2)
        
        logger.info(f"Saved processed file: {out_path} with {len(feature_cols)} features")
    
    logger.info("Enhanced processing completed successfully")


if __name__ == '__main__':
    process_all()
