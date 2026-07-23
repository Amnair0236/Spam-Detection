import joblib

def save_best_model(
        models,
        results,
        vectorizer,
        model_path="models/spam_model.pkl",
        vectorizer_path="models/tfidf_vectorizer.pkl"
):
    #Selects the model with the highest F1 Score and saves it.

    best_model_name = None
    best_model = None
    best_score = 0

    for model_name in models:
        score = results[model_name]["f1_score"]

        if score > best_score:
            best_score = score
            best_model_name = model_name
            best_model = models[model_name]

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    return best_model_name, best_score
