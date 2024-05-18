import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your dataset (replace "your_dataset.csv" with your dataset file)
data = pd.read_csv("malicious_phish.csv")
data=data.iloc[1:10000]
# Preprocessing
X = data['url']  # Use the 'url' column as the feature
y = data['type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF vectorizer for text data
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create and train a Support Vector Machine (SVM) model
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(tfidf_vectorizer.transform(['www.google.com']))
print(y_pred)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy}")
# print("Confusion Matrix:")
# print(conf_matrix)
# print("Classification Report:")
# print(classification_rep)

# You can now use this trained SVM model for phishing website detection.
# Be sure to replace "your_dataset.csv" with the actual path to your dataset file.