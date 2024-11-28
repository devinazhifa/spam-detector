import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("email_classification.csv")

# Preprocessing
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Convert labels to numeric
X = data['email']
y = data['label']

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'spam_detector.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
