import time

from auth.fyers_auth import fyers
from data.fetcher import fetch_candle_data
from data.processor import DataProcessor
from realtime.data_handler import subscribe_market_data  # To be implemented
from realtime.order_update_handler import subscribe_order_updates  # To be implemented

active_order_sleep = 2

def main():
    print("Starting Algo Trading Bot (Development Mode)...")

    # Initialize WebSocket subscriptions (market data, order updates)
    subscribe_market_data()
    subscribe_order_updates()

    # Fetch recent data once
    df = fetch_candle_data(
        days_back=5,
        index_symbol="NSE:NIFTY50-INDEX",
        interval_minutes=5,
        fyers=fyers,
        active_order_sleep=active_order_sleep,
    )

    # Process data
    processor = DataProcessor(df)
    processor.preprocess_datetime().clean_data().add_indicators()
    processed_df = processor.get_df()

    print(processed_df.head())

    # TODO: Run strategies on processed_df
    # signal = InsideCandleStrategy().generate_signal(processed_df)
    # or
    # signal = EMACrossoverStrategy().generate_signal(processed_df)
    # or
    # signal = MLModelStrategy().predict(processed_df)

    # TODO: Based on signal, place/manage orders
    # place_order(signal)
    # manage_orders()

if __name__ == "__main__":
    main()
