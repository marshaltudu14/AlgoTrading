import pytest
from src.config.instrument import Instrument

def test_instrument_creation():
    instrument = Instrument("Bank_Nifty", "OPTION", 25, 0.05)
    assert instrument.symbol == "Bank_Nifty"
    assert instrument.type == "OPTION"
    assert instrument.lot_size == 25
    assert instrument.tick_size == 0.05

def test_instrument_repr():
    instrument = Instrument("Nifty", "OPTION", 50, 0.05)
    assert repr(instrument) == "Instrument(symbol='Nifty', type='OPTION', lot_size=50, tick_size=0.05)"
