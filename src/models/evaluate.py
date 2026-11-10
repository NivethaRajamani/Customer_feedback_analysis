import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/processed/cleaned_reviews.csv")

model = joblib.load("models/model.pkl")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

X = vectorizer.transform(df["cleaned review"])

y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

labels = ["Negative", "Neutral", "Positive"]

cm = confusion_matrix(y_test, predictions, labels=labels)

cm_table = pd.DataFrame(cm, index=["actual_negative", "actual_neutral", "actual_positive"],
    columns=["pred_negative", "pred_neutral", "pred_positive"])


print("\nConfusion Matrix (rows = actual, columns = predicted)")
print(cm_table)


