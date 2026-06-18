from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

company_model = joblib.load("company_model.pkl")
location_model = joblib.load("location_model.pkl")
duration_model = joblib.load("duration_model.pkl")
stipend_model = joblib.load("stipend_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    job_title = request.form["job_title"]

    company = company_model.predict([job_title])[0]
    location = location_model.predict([job_title])[0]
    duration = duration_model.predict([job_title])[0]
    stipend = stipend_model.predict([job_title])[0]

    return render_template(
        "index.html",
        company=company,
        location=location,
        duration=duration,
        stipend=stipend
    )

if __name__ == "__main__":
    app.run(debug=True)