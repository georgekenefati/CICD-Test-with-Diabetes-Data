import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes


def load_data():
    """
    Load data using the load_diabetes function, create a DataFrame df, and return it.
    """
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def basic_statistics(df):
    """
    Calculate basic statistics using the describe function, create a DataFrame df_stats, and return it.
    """
    df_stats = df.describe()
    return df_stats


def plot_data(df):
    """
    Plot data using the plot function, create a DataFrame df_plot, and return it.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df["bmi"], df["target"], alpha=0.5)
    plt.xlabel("Body Mass Index (BMI)")
    plt.ylabel("Diabetes Progression")
    plt.title("Diabetes Progression vs. BMI")
    plt.savefig("diabetes_progression_vs_bmi.png")
    return plt


if __name__ == "__main__":
    df = load_data()
    print("Basic Statistics:\n", basic_statistics(df))
    plot_data(df)
    print("Plot saved to 'diabetes_progression_vs_bmi.png'.")
