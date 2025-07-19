import unittest
import pandas as pd
from pathlib import Path
import shutil
import os
from src.utils.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.test_data_final_dir = Path("test_data_final")
        self.test_data_final_dir.mkdir(exist_ok=True)
        self.test_data_raw_dir = Path("test_data_raw")
        self.test_data_raw_dir.mkdir(exist_ok=True)

    def tearDown(self):
        if self.test_data_final_dir.exists():
            shutil.rmtree(self.test_data_final_dir)
        if self.test_data_raw_dir.exists():
            shutil.rmtree(self.test_data_raw_dir)

    def _create_final_csv(self, filename, content):
        file_path = self.test_data_final_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path

    def _create_raw_csv(self, filename, content):
        file_path = self.test_data_raw_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path

    def test_load_all_processed_data_success(self):
        self._create_final_csv("data1.csv", "col1,col2\n1,A\n2,B")
        self._create_final_csv("data2.csv", "col1,col2\n3,C\n4,D")

        loader = DataLoader(final_data_dir=str(self.test_data_final_dir))
        df = loader.load_all_processed_data()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4)
        self.assertListEqual(df.columns.tolist(), ["col1", "col2"])
        self.assertListEqual(df["col1"].tolist(), [1, 2, 3, 4])

    def test_load_all_processed_data_empty_directory(self):
        loader = DataLoader(final_data_dir=str(self.test_data_final_dir))
        df = loader.load_all_processed_data()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_load_all_processed_data_corrupted_file(self):
        self._create_final_csv("data1.csv", "col1,col2\n1,A")
        self._create_final_csv("corrupted.csv", "this is not a csv file at all") # Malformed CSV
        self._create_final_csv("data2.csv", "col1,col2\n2,B")

        loader = DataLoader(final_data_dir=str(self.test_data_final_dir))
        df = loader.load_all_processed_data()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2) # Should load data1 and data2, skip corrupted
        self.assertListEqual(df["col1"].tolist(), [1, 2])

    def test_load_all_processed_data_non_existent_directory(self):
        loader = DataLoader(final_data_dir="non_existent_dir")
        df = loader.load_all_processed_data()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_load_all_processed_data_duplicate_rows(self):
        self._create_final_csv("data1.csv", "col1,col2\n1,A\n2,B")
        self._create_final_csv("data2.csv", "col1,col2\n1,A\n3,C")

        loader = DataLoader(final_data_dir=str(self.test_data_final_dir))
        df = loader.load_all_processed_data()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3) # 1,A should be deduplicated
        self.assertListEqual(sorted(df["col1"].tolist()), [1, 2, 3])

    def test_load_raw_data_for_symbol_success(self):
        self._create_raw_csv("Nifty_2.csv", "datetime,open,high,low,close\n2023-01-01,100,110,90,105\n2023-01-02,105,115,95,110")
        loader = DataLoader(final_data_dir=str(self.test_data_final_dir), raw_data_dir=str(self.test_data_raw_dir))
        df = loader.load_raw_data_for_symbol("Nifty_2")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertListEqual(df.columns.tolist(), ["datetime", "open", "high", "low", "close"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['datetime']))

    def test_load_raw_data_for_symbol_file_not_found(self):
        loader = DataLoader(final_data_dir=str(self.test_data_final_dir), raw_data_dir=str(self.test_data_raw_dir))
        df = loader.load_raw_data_for_symbol("NonExistentSymbol")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_load_raw_data_for_symbol_invalid_ohlc(self):
        self._create_raw_csv("Invalid_OHLC.csv", "datetime,open,high,low,close\n2023-01-01,100,90,110,105") # high < low
        loader = DataLoader(final_data_dir=str(self.test_data_final_dir), raw_data_dir=str(self.test_data_raw_dir))
        df = loader.load_raw_data_for_symbol("Invalid_OHLC")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_load_raw_data_for_symbol_missing_columns(self):
        self._create_raw_csv("Missing_Cols.csv", "datetime,open,high,close\n2023-01-01,100,110,105") # Missing low
        loader = DataLoader(final_data_dir=str(self.test_data_final_dir), raw_data_dir=str(self.test_data_raw_dir))
        df = loader.load_raw_data_for_symbol("Missing_Cols")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

if __name__ == '__main__':
    unittest.main()
