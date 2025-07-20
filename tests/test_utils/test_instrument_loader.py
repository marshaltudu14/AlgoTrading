import pytest
import os
from src.utils.instrument_loader import load_instruments
from src.config.instrument import Instrument

# Define the path to the instruments.yaml for testing
TEST_YAML_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'instruments.yaml')

def test_load_instruments_success():
    instruments = load_instruments(TEST_YAML_PATH)
    assert "Bank_Nifty" in instruments
    assert "Nifty" in instruments

    bank_nifty = instruments["Bank_Nifty"]
    assert isinstance(bank_nifty, Instrument)
    assert bank_nifty.symbol == "Bank_Nifty"
    assert bank_nifty.type == "OPTION"
    assert bank_nifty.lot_size == 25
    assert bank_nifty.tick_size == 0.05

    nifty = instruments["Nifty"]
    assert isinstance(nifty, Instrument)
    assert nifty.symbol == "Nifty"
    assert nifty.type == "OPTION"
    assert nifty.lot_size == 50
    assert nifty.tick_size == 0.05

def test_load_instruments_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_instruments("non_existent_file.yaml")

def test_load_instruments_invalid_yaml():
    # Create a temporary invalid YAML file for testing
    invalid_yaml_content = "instruments:\n  - symbol: Bank_Nifty\n    type: OPTION\n    lot_size: 25\n    # Missing tick_size intentionally\n"
    invalid_yaml_path = "invalid_instruments.yaml"
    with open(invalid_yaml_path, 'w') as f:
        f.write(invalid_yaml_content)

    with pytest.raises(KeyError):
        load_instruments(invalid_yaml_path)

    os.remove(invalid_yaml_path) # Clean up the temporary file
