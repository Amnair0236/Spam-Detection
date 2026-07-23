from src.train_model import (
    load_and_prepare_data,
    train_naive_bayes,
    train_logistic_regression,
    train_svm
)

from src.evaluate_model import evaluate_model
from src.model_selection import save_best_model

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

X_train, X_test, y_train, y_test, vectorizer = load_and_prepare_data()

#----------------------
# Naive Bayes
#----------------------

print("\nTraining Naive Bayes Model...")
nb_model = train_naive_bayes(X_train, y_train)

nb_results = evaluate_model(nb_model, X_test, y_test)

print("\nNaive Bayes Results")
print("-" * 40)

print(f"Accuracy : {nb_results['accuracy']:.4f}")
print(f"Precision : {nb_results['precision']:.4f}")
print(f"Recall : {nb_results['recall']:.4f}")
print(f"F1 Score : {nb_results['f1_score']:.4f}")

print("\nConfusion Matrix: ")
print(nb_results["confusion_matrix"])

print("\nClassification Report: ")
print(nb_results["classification_report"])

#--------------------------
# Logistic Regression
#--------------------------

print("\n" + "=" * 60)
print("Training Logistic Regression Model")
print("=" * 60)

lr_model = train_logistic_regression(X_train, y_train)

lr_results = evaluate_model(lr_model, X_test, y_test)

print("\nLogistic Regression Results")
print("-" * 40)

print(f"Accuracy : {lr_results['accuracy']:.4f}")
print(f"Precision: {lr_results['precision']:.4f}")
print(f"Recall: {lr_results['recall']:.4f}")
print(f"F1 Score: {lr_results['f1_score']:.4f}")

print("\nConfusion Matrix: ")
print(lr_results["confusion_matrix"])

print("\nClassification Results: ")
print(lr_results["classification_report"])

print("\n" + "=" * 60)
print("Training Support Vector Machine")
print("=" * 60)

svm_model = train_svm(X_train, y_train)

svm_results = evaluate_model(svm_model, X_test, y_test)

print("\nSupport Vector Machine Results")
print("-" * 40)

print(f"Accuracy: {svm_results['accuracy']:.4f}")
print(f"Precision: {svm_results['precision']:.4f}")
print(f"Recall: {svm_results['recall']:.4f}")
print(f"F1 Score: {svm_results['f1_score']:.4f}")

print("\nConfusion Matrix: ")
print(svm_results["confusion_matrix"])

print("\nClassification Report: ")
print(svm_results["classification_report"])

print("\n" + "=" * 60)
print("Selecting Best Model")
print("=" * 60)

models = {
    "Naive Bayes": nb_model,
    "Logistic Regression": lr_model,
    "Support Vector Machine": svm_model
}

results = {
    "Naive Bayes": nb_results,
    "Logistic Regression": lr_results,
    "Support Vector Machine": svm_results
}

best_model_name, best_score = save_best_model(models, results, vectorizer)

print(f"\nBest Model : {best_model_name}")
print(f"Best F1 Score : {best_score:.4f}")

print("\nModel Saved Successfully!")
print("Location: models/spam_model.pkl")

print("\nVectorizer saved successfully!")
print("Location: models/tfidf_vectorizer.pkl")