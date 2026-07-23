import joblib

from src.predict import predict_message

model  = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("=" * 60)
print("Spam Detection Prediction Test")
print("=" * 60)

message = input("\nEnter a message:\n\n")
label, confidence = predict_message(message, model, vectorizer)

print("\nPrediction: ", label)
print(f"Confidence: {confidence * 100:.2f}%")
