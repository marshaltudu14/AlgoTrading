import os
import pandas as pd
import pytest
from pathlib import Path
import shutil
import sys

# Add project root to the Python path to allow importing from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.test_data_generator import create_test_data_files
from src.config.settings import get_settings

@pytest.fixture(scope="module")
def test_data_directory():
    """Create a temporary directory for test data and clean it up afterward."""
    settings = get_settings()
    paths_config = settings.get('paths', {})
    test_dir = Path(paths_config.get('test_data_dir', 'data/test'))
    
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    yield str(test_dir)

def test_pipeline_runs_for_all_configured_test_symbols(test_data_directory):
    """
    Tests that the create_test_data_files function runs for all symbols
    defined in the configuration and creates output files for them.
    """
    # Arrange: Load the test symbols from the central config
    settings = get_settings()
    test_config = settings.get('testing_overrides', {}).get('test_data', {})
    symbols_to_test = test_config.get('symbols', [])
    assert symbols_to_test, "No test symbols found in config: testing_overrides.test_data.symbols"

    # Act: Run the data generation function
    created_files = create_test_data_files(
        data_dir=test_data_directory,
        num_rows=200,
        create_multiple_instruments=True
    )

    # Assert: Check that a features file was created for each symbol
    for symbol in symbols_to_test:
        features_file_path_str = created_files.get(f'features_{symbol}')
        assert features_file_path_str is not None, f"Function should return a path for the features file of {symbol}."
        
        features_file_path = Path(features_file_path_str)
        assert features_file_path.exists(), f"The features file for {symbol} was not actually created at {features_file_path}"
