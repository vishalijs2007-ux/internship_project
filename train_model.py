import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load Dataset
df = pd.read_csv("internship.csv")

# Remove missing values
df = df.dropna()

# Input Feature
X = df["internship_title"]

# Output Labels
y_company = df["company_name"]
y_location = df["location"]
y_duration = df["duration"]
y_stipend = df["stipend"]

# Model Builder
def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=1000))
    ])

# Train Models
company_model = build_model()
company_model.fit(X, y_company)

location_model = build_model()
location_model.fit(X, y_location)

duration_model = build_model()
duration_model.fit(X, y_duration)

stipend_model = build_model()
stipend_model.fit(X, y_stipend)

# Save Models
joblib.dump(company_model, "company_model.pkl")
joblib.dump(location_model, "location_model.pkl")
joblib.dump(duration_model, "duration_model.pkl")
joblib.dump(stipend_model, "stipend_model.pkl")

print("All models trained successfully!")