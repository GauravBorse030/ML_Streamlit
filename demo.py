# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

# Load dataset
df = pd.read_csv("spam.csv")

# Features and target
X = df['text']
y = df['label_num']

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "svm_spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
