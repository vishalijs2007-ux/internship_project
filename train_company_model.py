import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("internship.csv")

# Input and Output columns
X = df["internship_title"]
y_company = df["company_name"]

# Create model
company_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
company_model.fit(X, y_company)

# Save model
joblib.dump(company_model, "company_model.pkl")

print("company_model.pkl created successfully!")