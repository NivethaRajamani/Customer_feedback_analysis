import joblib
from src.data.preprocess import preprocess_text
from pydantic import BaseModel
from fastapi import FastAPI
from src.alerts.telegram_alert import send_negative_alert


app = FastAPI()

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

class Review(BaseModel) :
    text :str


@app.post("/predict")
def predict(review : Review) :

    vectorized = vectorizer.transform([preprocess_text(review.text)])
    
    sentiment = model.predict(vectorized)[0]

    if sentiment.lower() == "negative":
        send_negative_alert(review.text)
    
    return {
                "sentiment" : sentiment
            }




