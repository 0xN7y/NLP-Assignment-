import nltk
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pandas as pd
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
nlp = spacy.load("en_core_web_sm")


text = "Natural Language Processing is a fascinating field. It combines linguistics and computer science!"

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return cleaned_tokens


cleaned_tokens = preprocess(text)
print("Cleaned Tokens:", cleaned_tokens)


bigrams = list(ngrams(cleaned_tokens, 2))
print("Bigrams:", bigrams)

sentence = "Barack Obama was born in Hawaii and was elected president in 2008."
doc = nlp(sentence)

print("Named Entities:")
for ent in doc.ents:
    print(ent.text, ent.label_)

sentences = [
    "I love machine learning.",
    "Natural language processing is a part of AI.",
    "AI is the future."
]

count_vec = CountVectorizer()
X_count = count_vec.fit_transform(sentences)
print("Count Vectorizer Output:\n", X_count.toarray())
print("Count Vectorizer Feature Names:\n", count_vec.get_feature_names_out())

tfidf_vec = TfidfVectorizer()
X_tfidf = tfidf_vec.fit_transform(sentences)
print("\nTF-IDF Vectorizer Output:\n", X_tfidf.toarray())
print("TF-IDF Feature Names:\n", tfidf_vec.get_feature_names_out())
