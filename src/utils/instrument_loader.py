import yaml
from src.config.instrument import Instrument

def load_instruments(file_path: str) -> dict[str, Instrument]:
    """
    Loads instrument data from a YAML file and parses it into a dictionary of Instrument objects.
    """
    instruments = {}
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
        for item in data.get('instruments', []):
            symbol = item['symbol']
            instrument_type = item['type']
            lot_size = item['lot_size']
            tick_size = item['tick_size']
            instruments[symbol] = Instrument(symbol, instrument_type, lot_size, tick_size)
    return instruments

if __name__ == '__main__':
    # Example usage:
    # Assuming instruments.yaml is in the config directory relative to the project root
    # You might need to adjust the path based on where this script is run from
    try:
        loaded_instruments = load_instruments('../../config/instruments.yaml')
        for symbol, instrument in loaded_instruments.items():
            print(f"Loaded Instrument: {instrument}")
    except FileNotFoundError:
        print("Error: instruments.yaml not found. Make sure the path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")