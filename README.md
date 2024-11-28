
# Real-Time Spam Detector Using Machine Learning

This project implements a **Spam Detection System** using machine learning in a **server-client architecture** with **ZeroMQ** for communication. It uses **Multinomial Naive Bayes** for spam classification and provides a simple web interface built with Flask for real-time spam detection.

## Requirements
Before running the project, make sure to install the following dependencies:

Python Libraries:
- **Flask** - For creating the web application.
- **scikit-learn** - For machine learning algorithms and model handling.
- **zmq** - For implementing the server-client communication.
- **joblib** - For loading pre-trained machine learning models.
- **nltk** - For natural language processing.

To install the required libraries, you can run:

```bash
pip install flask scikit-learn zmq joblib nltk
```

## Steps to Run the Project
**1. Clone the Repository**
```bash
git clone https://github.com/devinazhifa/spam-detector.git
cd spam-detector
```

**2. Start the Server**
```bash
python server.py
```

**3. Start the Client**
```bash
python client.py
```

**4. Start the Web Interface**
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.
