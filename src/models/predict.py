import joblib
from src.data.preprocess import preprocess_text

model = joblib.load("models/model.pkl")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

while True :

    review = input("Enter your review : ")

    if review.lower() == "exit" :

        break

    else :
        review = [preprocess_text(review)]

        review = vectorizer.transform(review)

        prediction = model.predict(review)


        print(prediction[0])


