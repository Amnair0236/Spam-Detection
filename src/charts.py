"""
charts.py

This module generates professional charts 
for the SMS Spam Detection dashboard.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Load Model Comparison Data

def load_comparison_data():
    """
    Load model comparison results.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv("reports/model_comparison.csv")

# Generic Chart Creator

def create_metric_chart(metric):
    """
    Create a professional bar chart for a model metric.

    Parameters
    ----------
    metric : str
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    comparison = load_comparison_data()
    fig, ax = plt.subplots(figsize=(8,5))

    bars = ax.bar(
        comparison["Model"],
        comparison[metric],
        width=0.55
    )

    ax.set_title(
        f"{metric} Comparison",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Machine Learning Models", fontsize=11)

    ax.set_ylabel(metric, fontsize=11)

    ax.set_ylim(0, 1.05)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.set_axisbelow(True)

    #Display percentage above each bar
    for bar in bars:

        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height*100:.2f}%",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )
    plt.tight_layout()

    return fig

# Individual Charts

def create_accuracy_chart():
    return create_metric_chart("Accuracy")

def create_precision_chart():
    return create_metric_chart("Precision")

def create_recall_chart():
    return create_metric_chart("Recall")

def create_f1_chart():
    return create_metric_chart("F1 Score")

# Dataset Distribution

def create_dataset_distribution():
    labels = ["Ham", "Spam"]
    sizes = [4825, 747]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Dataset Distribution", fontsize=15, fontweight="bold")

    plt.tight_layout()

    return fig

