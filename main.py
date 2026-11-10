from src.data.load_data import load_data
from src.data.preprocess import create_sentiment_labels
from src.data.preprocess import preprocess_text
from src.data.process_dataset import process_dataset


process_dataset()



# def test_load_data():

#     return load_data(r"D:\AI Journey\AI\NLP\Practice_Projects\customer_feedback_analysis\data\raw\Reviews.csv")

# df = test_load_data()
# #print(df.head())
# df = df.sample(1000, random_state=42)   # <- add this

# df["sentiment"] = df["Text"].apply(create_sentiment_labels)

# print(df[["Text", "sentiment"]].head())


# print(preprocess_text("""
# This product is AMAZING!!!
# I bought 2 units. Visit https://example.com for more info!"""))



