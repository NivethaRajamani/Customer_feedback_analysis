import pandas as pd
import re
import spacy
from src.data.preprocess import create_sentiment_labels, preprocess_batch

def process_dataset():

    print("Loading dataset...")
    df = pd.read_csv("data/raw/Reviews.csv")
    print("Dataset Loaded")

    df = df.sample(10000, random_state = 42)

    df["sentiment"]= df["Text"].apply(create_sentiment_labels)


    print("Created Sentiment label")

    text = df["Text"].fillna("").astype(str).str.lower()
    text = text.str.replace(r"https?://\S+", "", regex=True)
    text = text.str.replace(r"[^a-zA-Z\s]", "", regex = True)


    print("Did basic cleaning")

    df["cleaned review"] = preprocess_batch(text)
    

    print("preprocessing completed")

    df.to_csv("data/processed/cleaned_reviews.csv", index = False)


    print("File_saved")




