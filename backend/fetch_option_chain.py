#!/usr/bin/env python3
"""
Fetch Option Chain Data Script

This script fetches option chain data from Fyers API for a given symbol,
including bid/ask prices, open interest, PCR, and VIX data.

Usage:
    python fetch_option_chain.py --symbol <SYMBOL> [--strikecount <STRIKE_COUNT>]

Example:
    python fetch_option_chain.py --symbol NSE:NIFTY50-INDEX --strikecount 5
"""

import argparse
import json
import os
import requests
import yaml
import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fyers_apiv3 import fyersModel

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.fyers_auth_service import authenticate_fyers_user, create_fyers_model, FyersAuthenticationError
from config.fyers_config import (
    APP_ID, SECRET_KEY, REDIRECT_URI, FYERS_USER, FYERS_PIN, FYERS_TOTP
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration from config files."""
    config = {}

    # Load instruments configuration
    try:
        with open('backend/config/instruments.yaml', 'r') as f:
            config['instruments'] = yaml.safe_load(f)
    except FileNotFoundError:
        print("Warning: config/instruments.yaml not found")
        config['instruments'] = {}

    return config

async def get_access_token_async() -> Optional[str]:
    """Get Fyers access token using the authentication service."""
    try:
        access_token = await authenticate_fyers_user(
            app_id=APP_ID,
            secret_key=SECRET_KEY,
            redirect_uri=REDIRECT_URI,
            fy_id=FYERS_USER,
            pin=FYERS_PIN,
            totp_secret=FYERS_TOTP
        )
        return access_token
    except FyersAuthenticationError as e:
        logger.error(f"Authentication error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error getting access token: {e}")
        return None

async def fetch_option_chain_data(symbol: str, strike_count: int = 5) -> Optional[Dict[str, Any]]:
    """Fetch option chain data from Fyers API."""
    try:
        # Get access token using the authentication service
        access_token = await get_access_token_async()
        if not access_token:
            logger.error("Failed to get access token")
            return None

        # Create Fyers model instance
        fyers = create_fyers_model(access_token, APP_ID)

        # Prepare API request
        url = f"https://api-t1.fyers.in/data/options-chain-v3?symbol={symbol}&strikecount={strike_count}"
        headers = {
            'Authorization': access_token,
            'Content-Type': 'application/json'
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            if data.get('code') == 200:
                return data.get('data')
            else:
                logger.error(f"Error fetching option chain data: {data.get('message', 'Unknown error')}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Error making API request: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            return None

    except Exception as e:
        logger.error(f"Unexpected error in fetch_option_chain_data: {e}")
        return None

def calculate_pcr(option_chain_data: Dict[str, Any]) -> float:
    """Calculate Put-Call Ratio from option chain data."""
    options_chain = option_chain_data.get('optionsChain', [])

    total_call_oi = 0
    total_put_oi = 0

    for option in options_chain:
        if option.get('option_type') == 'CE':
            total_call_oi += option.get('oi', 0)
        elif option.get('option_type') == 'PE':
            total_put_oi += option.get('oi', 0)

    if total_call_oi > 0:
        return total_put_oi / total_call_oi
    else:
        return 0.0

def analyze_option_chain(option_chain_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze option chain data and extract key market insights."""
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'vix_data': option_chain_data.get('indiavixData', {}),
        'pcr_ratio': calculate_pcr(option_chain_data),
        'total_call_oi': 0,
        'total_put_oi': 0,
        'max_oi_call': None,
        'max_oi_put': None,
        'liquidity_score': 0,
        'key_levels': {
            'support': [],
            'resistance': []
        },
        'oi_distribution': {
            'call_oi_by_strike': {},
            'put_oi_by_strike': {},
            'top_5_call_oi': [],
            'top_5_put_oi': [],
            'oi_concentration_zones': []
        }
    }

    options_chain = option_chain_data.get('optionsChain', [])

    # Calculate OI metrics
    max_call_oi = 0
    max_put_oi = 0
    total_spread = 0
    liquid_options = 0

    # Collect OI data by strike price
    call_oi_data = []
    put_oi_data = []

    for option in options_chain:
        oi = option.get('oi', 0)
        option_type = option.get('option_type')
        strike = option.get('strike_price')

        if option_type == 'CE':
            analysis['total_call_oi'] += oi
            call_oi_data.append({'strike': strike, 'oi': oi, 'ltp': option.get('ltp')})
            if oi > max_call_oi:
                max_call_oi = oi
                analysis['max_oi_call'] = {
                    'strike': strike,
                    'oi': oi,
                    'ltp': option.get('ltp')
                }
        elif option_type == 'PE':
            analysis['total_put_oi'] += oi
            put_oi_data.append({'strike': strike, 'oi': oi, 'ltp': option.get('ltp')})
            if oi > max_put_oi:
                max_put_oi = oi
                analysis['max_oi_put'] = {
                    'strike': strike,
                    'oi': oi,
                    'ltp': option.get('ltp')
                }

        # Calculate liquidity score based on bid-ask spread
        bid = option.get('bid', 0)
        ask = option.get('ask', 0)
        if bid > 0 and ask > 0:
            spread = ((ask - bid) / ask) * 100
            total_spread += spread
            liquid_options += 1

    if liquid_options > 0:
        analysis['liquidity_score'] = total_spread / liquid_options

    # Sort OI data and get top concentrations
    call_oi_data.sort(key=lambda x: x['oi'], reverse=True)
    put_oi_data.sort(key=lambda x: x['oi'], reverse=True)

    # Get top 5 OI concentrations for calls and puts
    analysis['oi_distribution']['top_5_call_oi'] = call_oi_data[:5]
    analysis['oi_distribution']['top_5_put_oi'] = put_oi_data[:5]

    # Create OI dictionaries by strike
    for item in call_oi_data:
        analysis['oi_distribution']['call_oi_by_strike'][item['strike']] = item['oi']
    for item in put_oi_data:
        analysis['oi_distribution']['put_oi_by_strike'][item['strike']] = item['oi']

    # Identify OI concentration zones (areas with high OI on both sides)
    identify_oi_concentration_zones(analysis)

    # Identify key levels based on OI concentrations
    if analysis['max_oi_put']:
        analysis['key_levels']['support'].append(analysis['max_oi_put']['strike'])
    if analysis['max_oi_call']:
        analysis['key_levels']['resistance'].append(analysis['max_oi_call']['strike'])

    return analysis

def identify_oi_concentration_zones(analysis: Dict[str, Any]):
    """Identify zones with high OI concentrations on both call and put sides."""
    call_oi = analysis['oi_distribution']['call_oi_by_strike']
    put_oi = analysis['oi_distribution']['put_oi_by_strike']

    # Find strikes with significant OI on both sides
    all_strikes = set(call_oi.keys()) | set(put_oi.keys())
    concentration_zones = []

    for strike in sorted(all_strikes):
        call_oi_at_strike = call_oi.get(strike, 0)
        put_oi_at_strike = put_oi.get(strike, 0)

        # Define threshold for significant OI (top 20% of max OI)
        max_call_oi = max(call_oi.values()) if call_oi else 0
        max_put_oi = max(put_oi.values()) if put_oi else 0

        call_threshold = max_call_oi * 0.2
        put_threshold = max_put_oi * 0.2

        if call_oi_at_strike >= call_threshold or put_oi_at_strike >= put_threshold:
            concentration_zones.append({
                'strike': strike,
                'call_oi': call_oi_at_strike,
                'put_oi': put_oi_at_strike,
                'total_oi': call_oi_at_strike + put_oi_at_strike,
                'oi_type': 'call' if call_oi_at_strike > put_oi_at_strike else 'put' if put_oi_at_strike > call_oi_at_strike else 'balanced'
            })

    analysis['oi_distribution']['oi_concentration_zones'] = concentration_zones

def save_option_chain_data(data: Dict[str, Any], symbol: str):
    """Save essential option chain analysis data to JSON file."""
    # Create data directory if it doesn't exist
    os.makedirs('backend/data', exist_ok=True)

    # Only save essential analysis data, not the full option chain
    analysis_data = {
        'symbol': symbol,
        'fetched_at': datetime.now().isoformat(),
        'analysis': data
    }

    # Save to file
    filename = 'backend/data/option_chain_data.json'
    try:
        with open(filename, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"Option chain analysis saved to {filename}")
    except Exception as e:
        print(f"Error saving option chain data: {e}")

def print_basic_summary(analysis: Dict[str, Any]):
    """Print basic summary information."""
    # VIX data
    vix_data = analysis.get('vix_data', {})
    if vix_data:
        print(f"VIX: {vix_data.get('ltp', 'N/A')} ({vix_data.get('ltpchp', 'N/A')}%)")

    # PCR ratio
    pcr = analysis.get('pcr_ratio', 0)
    print(f"Put-Call Ratio: {pcr:.2f}")

    # Market sentiment based on PCR
    if pcr > 1.5:
        sentiment = "Bearish (High PCR)"
    elif pcr > 1.0:
        sentiment = "Neutral to Slightly Bearish"
    elif pcr > 0.7:
        sentiment = "Neutral"
    else:
        sentiment = "Bullish (Low PCR)"
    print(f"Market Sentiment: {sentiment}")

    # Key levels
    key_levels = analysis.get('key_levels', {})
    if key_levels.get('support'):
        print(f"Support Levels: {', '.join(map(str, key_levels['support']))}")
    if key_levels.get('resistance'):
        print(f"Resistance Levels: {', '.join(map(str, key_levels['resistance']))}")

async def main():
    """Main function to fetch option chain data."""
    parser = argparse.ArgumentParser(description='Fetch option chain data from Fyers API')
    parser.add_argument('--symbol', required=True, help='Symbol to fetch option chain for')
    parser.add_argument('--strikecount', type=int, default=5, help='Number of strikes to fetch (default: 5)')
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Fetch option chain data
    print(f"Fetching option chain data for {args.symbol}...")
    option_chain_data = await fetch_option_chain_data(args.symbol, args.strikecount)

    if option_chain_data:
        # Analyze the data
        analysis = analyze_option_chain(option_chain_data)

        # Save the analysis data
        save_option_chain_data(analysis, args.symbol)

        # Print basic summary
        print_basic_summary(analysis)

        print(f"\nOption chain data successfully fetched and saved to backend/data/option_chain_data.json")
    else:
        print("Failed to fetch option chain data")

if __name__ == "__main__":
    asyncio.run(main())