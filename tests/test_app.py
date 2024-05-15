import unittest
import pandas as pd

from app import load_data, basic_statistics, plot_data


class TestApp(unittest.TestCase):
    def test_load_data(self):
        df = load_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (442, 11))  # Shape of the dataset

    def test_basic_statistics(self):
        df = load_data()
        df_stats = basic_statistics(df)
        self.assertIsInstance(df_stats, pd.DataFrame)
        self.assertEqual(df_stats.shape, (8, 11))

    def test_plot_data(self):
        df = load_data()
        plot_data(df)
        self.assertTrue("diabetes_progression_vs_bmi.png" in plot_data(df))


if __name__ == "__main__":
    unittest.main()
