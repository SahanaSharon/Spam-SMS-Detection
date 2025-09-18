import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import os
DATA_PATH = os.path.join("data", "spam_sample.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please place dataset at this path.")

df = pd.read_csv(DATA_PATH)

# Rename columns if needed
if list(df.columns) != ["label", "text"]:  
    df.columns = ["label", "text"]

# Handle missing values
df["text"] = df["text"].fillna("")
df["label"] = df["label"].fillna("ham")  # default to ham if missing

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2))),
    ("clf", MultinomialNB())
])

model.fit(X_train, y_train)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to", MODEL_PATH)
print("Accuracy on test set:", round(model.score(X_test, y_test) * 100, 2), "%")
