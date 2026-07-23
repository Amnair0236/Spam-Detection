from sklearn.feature_extraction.text import TfidfVectorizer

def create_vectorizer():
    #Creates and returns a TF-IDF vectorizer
    vectorizer = TfidfVectorizer (
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )
    return vectorizer

def fit_transform_text(vectorizer, text_data):
    #Learns the vocabulary and converts text into TF-IDF features.
    #Used only during training
    feature_matrix = vectorizer.fit_transform(text_data)
    return feature_matrix

def transform_text(vectorizer, text_data):
    #Converts new text into TF-IDF features using an existing vocabulary
    #Used during prediction
    
    feature_matrix = vectorizer.transform(text_data)
    return feature_matrix