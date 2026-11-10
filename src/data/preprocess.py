import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import string
import spacy

nlp = spacy.load("en_core_web_sm")

KEEP = ["not", "no", "never", "nor", "n't", "none"]

try:
     nltk.data.find("sentiment/vader_lexicon.zip")
except :
     nltk.download("vader_lexicon")


analyzer = SentimentIntensityAnalyzer()



def create_sentiment_labels(text) :

    score = analyzer.polarity_scores(str(text))["compound"]

    if score>=0.05:
        return "Positive"
    elif score<=-0.05:
        return "Negative"
    else:
        return "Neutral"   


def preprocess_text(text) :

    #stop_words = set(stopwords.words("english"))
    #lemmatizer = WordNetLemmatizer()

    text = str(text).lower()
    text = re.sub(r"https?://\S+","", text)
    #text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"[^a-zA-Z]", "", text)
    doc = nlp(text)
    words = []

    for token in doc :
        if token.lemma_ in KEEP or not token.is_stop :
            words.append(token.lemma_)


    return " ".join(words)



def preprocess_batch(texts) :

    cleaned = []

    docs = nlp.pipe(texts, disable = ["ner", "parser"], batch_size = 200)

    for doc in docs :

        words = [t.lemma_ for t in doc if t.lemma_ in KEEP or not t.is_stop]
        cleaned.append(" ".join(words))

    return cleaned





    # text = text.translate(str.maketrans("", "", string.punctuation))
    # tokens = text.split()
    # tokens = [word for word in tokens if word not in stop_words]
    # #tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # return " ".join(tokens)









