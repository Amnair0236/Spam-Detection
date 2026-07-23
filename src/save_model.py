import joblib

def save_model(model, vectorizer):

    joblib.dump(model, "models/spam_model.pkl")

    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    print("\nModel saved successfully!")
    print("Location: models/spam_model.pkl")

    print("\nVectorizer saved successfully!")
    print("Location: models/tfidf_vectorizer.pkl")
