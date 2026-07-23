import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.preprocess import clean_text
from src.feature_extraction import (
    create_vectorizer,
    fit_transform_text
)

def load_and_prepare_data():
    df = pd.read_csv(
        "data/spam.csv",
        encoding="latin-1"
    )

    df = df[['v1', 'v2']]
    df.columns = [
        "label",
        "message"
    ]

    df["clean_message"] = df["message"].apply(clean_text)

    df["label"] = df["label"].map(
        {
            "ham": 0,
            "spam": 1
        }
    )

    vectorizer = create_vectorizer()

    X = fit_transform_text(
        vectorizer,
        df["clean_message"]
    )

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        vectorizer
    )

def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_train, y_train)
    return model    


def train_svm(X_train, y_train):
    #Train Support Vector Machine

    model = SVC(
        kernel="linear",
        probability=True,
        random_state=42
    ) 
    model.fit(X_train, y_train)

    return model