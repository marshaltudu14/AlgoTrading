import pandas as pd
from pytz import timezone

class DataProcessor:
    """
    Process candle DataFrame: convert datetime to local timezone and clean volume column.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def preprocess_datetime(self):
        ist = timezone('Asia/Kolkata')
        # Convert Unix timestamp to datetime
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], unit='s')
        # Localize UTC then convert to IST and drop tz info
        self.df['datetime'] = (
            self.df['datetime']
                .dt.tz_localize('UTC')
                .dt.tz_convert(ist)
                .dt.tz_localize(None)
        )
        # Check duplicates or missing
        if self.df['datetime'].duplicated().any() or self.df['datetime'].isnull().any():
            raise ValueError("The 'datetime' column contains duplicates or missing values.")
        # Sort and set index
        self.df.sort_values('datetime', inplace=True)
        self.df.set_index('datetime', inplace=True)
        return self

    def clean_data(self):
        # Drop volume if zero or NaN
        if 'volume' in self.df.columns:
            if self.df['volume'].isnull().any() or (self.df['volume'] == 0).any():
                self.df.drop('volume', axis=1, inplace=True, errors='ignore')
        return self
