"""
utils.py

Utilities functions used throughout the Spam Detection project.
"""

import pandas as pd

# Load Dataset

def load_dataset():
    """Load the SMS Spam Collection dataset.
    
    Returns
    -------
    pandas.DataFrame
    """

    dataframe = pd.read_csv("data/spam.csv", encoding="latin-1")

    #Keep only required columns
    dataframe = dataframe.iloc[:, :2]

    dataframe.columns = ["Label", "Message"]

    return dataframe

# Dataset Statistics

def get_dataset_statistics():
    """Returns useful dataset statistics.
    
    Returns
    -------
    dict
    """

    dataframe = load_dataset()
    total_messages = len(dataframe)

    spam_messages = len(dataframe[dataframe["Label"] == "spam"])

    ham_messages = len(dataframe[dataframe["Label"] == "ham"])

    spam_percentage = round((spam_messages / total_messages) * 100, 2)
    ham_percentage = round((ham_messages / total_messages) * 100, 2)
    return {
        "Total Messages": total_messages,
        "Spam Messages": spam_messages,
        "Ham Messages": ham_messages,
        "Spam Percentage": spam_percentage,
        "Ham Percentage": ham_percentage
    }

# Load Model Comparison
def load_model_comparison():
    """Load model comparison CSV.
    
    Returns
    -------
    pandas.DataFrame
    """

    dataframe = pd.read_csv("reports/model_comparison.csv")

    return dataframe

# Best Model
def get_best_model():
    """
    Returns the best trained model.

    Returns
    -------
    str
    """
    dataframe = load_model_comparison()

    best_model = dataframe.sort_values(
        by="F1 Score",
        ascending=False
    ).iloc[0]["Model"]

    return best_model