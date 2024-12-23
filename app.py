from flask import Flask, render_template, request
import joblib
from dotenv import load_dotenv
import os
import replicate

# Load model and vectorizer
model = joblib.load('spam_detector.pkl')
vectorizer = joblib.load('vectorizer.pkl')

load_dotenv()
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

app = Flask(__name__)

def get_llm_advice_replicate(message, result):
    try:
        # Define the pre-prompt and user prompt
        pre_prompt = (
            "You are a helpful assistant. You provide advice based on spam classification. "
            "Do not respond as 'User' or pretend to be 'User'. You only respond as 'Assistant'."
        )
        user_prompt = f"How to deal with '{result}' messages? Provide brief suggestions on how to handle this"

        # Call the Replicate API
        output = replicate.run(
            "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
            input={
                "prompt": f"{pre_prompt} {user_prompt} Assistant:",
                "temperature": 0.1,
                "top_p": 0.9,
                "max_length": 128,
                "repetition_penalty": 1,
            },
        )

        # Combine the output if it's streamed
        full_response = "".join(output)
        return full_response
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

        # Get advice from LLM via Replicate API
        llm_advice = get_llm_advice_replicate(input_message, result)

    return render_template("index.html", result=result, input_message=input_message, llm_advice=llm_advice)

if __name__ == "__main__":
    app.run(debug=True)
