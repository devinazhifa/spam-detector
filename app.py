from flask import Flask, render_template, request
import joblib

# Load model and vectorizer
model = joblib.load('spam_detector.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_message = None
    if request.method == "POST":
        # Get message from the form
        input_message = request.form["message"]
        
        # Predict using the model
        X = vectorizer.transform([input_message])
        prediction = model.predict(X)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
    
    return render_template("index.html", result=result, input_message=input_message)

if __name__ == "__main__":
    app.run(debug=True)
