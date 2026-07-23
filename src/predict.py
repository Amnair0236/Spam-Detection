"""
predict.py

This module handles loading the trained model, preprocessing input messages, 
generating predictions, 
and returning structured prediction results.
"""
import joblib

from src.preprocess import clean_text
from src.explain import generate_explanation

# Load Saved Model and Vectorizer
MODEL_PATH = "models/spam_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# Predict Message
def predict_message(message):
    """
    Predict whether a message is Spam or Ham.

    Parameters
    ----------
    message : str
        User input message

    Returns
    -------
    dict{
        "label": str,
        "confidence": float,
        "confidence_level": str,
        "explanation": list
    }
    """

    # Clean Text
    cleaned_message = clean_text(message)

    # TF-IDF Transformation
    message_vector = vectorizer.transform([cleaned_message])

    # Prediction
    prediction = model.predict(message_vector)[0]

    probabilities = model.predict_proba(message_vector)[0]

    confidence = float(max(probabilities) * 100)

    # Convert Label
    label = "Spam" if prediction == 1 else "Ham"
    
    # Confidence Category
    if confidence >= 95:
        confidence_level = "Very High"
    elif confidence >= 85:
        confidence_level = "High"
    elif confidence >= 70:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"
    
    # Explanation
    explanation = generate_explanation(message, label)

    #Return Prediction Object
    return {
        "label": label,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "explanation": explanation
    }

