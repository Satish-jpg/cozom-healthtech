
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import joblib

# Load dataset (replace with your actual dataset path)
data = pd.DataFrame({
    'Symptom': ['fever', 'headache', 'chest pain', 'sore throat'],
    'Disease': ['Flu', 'Migraine', 'Heart Disease', 'Common Cold']
})

# Data preparation
X = data['Symptom']
y = data['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'disease_prediction_model.pkl')
