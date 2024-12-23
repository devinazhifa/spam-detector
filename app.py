from flask import Flask, render_template, request
import joblib
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load model and vectorizer
model = joblib.load('spam_detector.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Set OpenAI API key from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)

def get_llm_advice_openai(message, result):
    try:
        # Construct the prompt
        prompt = f"The message is classified as '{result}'. Provide advice on how to handle this in brief."
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can replace with other models like "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract response content
        advice = response['choices'][0]['message']['content']
        return advice
    except Exception as e:
        return f"Error fetching advice: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_message = None
    llm_advice = None
    if request.method == "POST":
        # Get message from the form
        input_message = request.form["message"]
        
        # Predict using the model
        X = vectorizer.transform([input_message])
        prediction = model.predict(X)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        
        # Get advice from OpenAI LLM
        llm_advice = get_llm_advice_openai(input_message, result)
    
    return render_template("index.html", result=result, input_message=input_message, llm_advice=llm_advice)

if __name__ == "__main__":
    app.run(debug=True)
