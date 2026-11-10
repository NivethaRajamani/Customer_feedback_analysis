import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

df = pd.read_csv("data/processed/cleaned_reviews.csv")

vectorizer = TfidfVectorizer(max_features=1000, ngram_range = (1,2))


print(df["cleaned review"].isna().sum())

print(df["cleaned review"].map(type).value_counts())

feature_extraction = vectorizer.fit_transform(df["cleaned review"])


model_file = joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print(feature_extraction.shape)
