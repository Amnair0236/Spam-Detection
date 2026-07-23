import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

stemmer = PorterStemmer()

stop_words = set(stopwords.words("english"))

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    words = text.split()

    cleaned_words = []
    
    for word in words:
        if word not in stop_words and len(word) > 2:
            stemmed_word = stemmer.stem(word)
            cleaned_words.append(stemmed_word)

    cleaned_text = " ".join(cleaned_words)
    return cleaned_text