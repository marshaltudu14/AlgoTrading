"""
Processes raw market data, including datetime conversion, cleaning, and indicator calculation.
"""

import pandas as pd
import pandas_ta as ta
from pytz import timezone

class DataProcessor:
    def __init__(self, df):
        self.df = df.copy()

    def preprocess_datetime(self):
        """
        Convert Unix timestamp to Asia/Kolkata timezone, check duplicates, sort, and set index.
        """
        ist = timezone('Asia/Kolkata')

        self.df['datetime'] = pd.to_datetime(self.df['datetime'], unit='s')
        self.df['datetime'] = (self.df['datetime']
                               .dt.tz_localize('UTC')
                               .dt.tz_convert(ist)
                               .dt.tz_localize(None))

        if self.df['datetime'].duplicated().any():
            print("Warning: Duplicate timestamps found. Dropping duplicates.")
            self.df = self.df.drop_duplicates(subset='datetime')

        if self.df['datetime'].isnull().any():
            print("Warning: Missing timestamps found. Dropping missing values.")
            self.df = self.df.dropna(subset=['datetime'])

        self.df.sort_values('datetime', inplace=True)
        self.df.set_index('datetime', inplace=True)

        return self

    def clean_data(self):
        """
        Remove volume column if missing or zero (common for indices).
        """
        if 'volume' in self.df.columns:
            if self.df['volume'].isnull().any() or (self.df['volume'] == 0).any():
                self.df.drop('volume', axis=1, inplace=True, errors='ignore')
        return self

    def add_indicators(self, ema_periods=None, atr_period=14):
        """
        Add EMA and ATR indicators.

        Args:
            ema_periods (list): List of EMA periods to calculate.
            atr_period (int): ATR period.
        """
        if ema_periods is None:
            ema_periods = [9, 21]

        for period in ema_periods:
            self.df[f'EMA_{period}'] = ta.ema(self.df['close'], length=period)

        self.df['ATR'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=atr_period)

        return self

    def get_df(self):
        return self.df
