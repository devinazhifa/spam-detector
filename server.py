import zmq
import joblib

# Load model and vectorizer
model = joblib.load('spam_detector.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# ZeroMQ setup
context = zmq.Context()
socket = context.socket(zmq.REP)  # REP for Reply
socket.bind("tcp://*:5555")

print("Server is running...")

while True:
    # Receive message from client
    message = socket.recv_string()
    print(f"Received: {message}")

    # Preprocess and predict
    X = vectorizer.transform([message])
    prediction = model.predict(X)[0]
    result = "Spam" if prediction == 1 else "Not Spam"

    # Send result back to client
    socket.send_string(result)
