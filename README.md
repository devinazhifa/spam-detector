
# Real-Time Spam Detector

This project implements a **Spam Detection System** using machine learning in a **server-client architecture** with **ZeroMQ** for communication. It uses **Multinomial Naive Bayes** for spam classification and integrates **Large Language Model (LLM)** from Replicate to provide additional insights and advice based on the classification results. The project also features a simple web interface for real-time spam detection.

## Requirements
Before running the project, make sure to install the following dependencies:

Python Libraries:
- **flask** - For creating the web application.
- **scikit-learn** - For machine learning algorithms and model handling.
- **zmq** - For implementing the server-client communication.
- **joblib** - For loading pre-trained machine learning models.
- **nltk** - For natural language processing.
- **requests** - For making API calls to the Replicate service.

To install the required libraries, you can run:

```bash
pip install flask scikit-learn zmq joblib nltk requests
```

## Steps to Run the Project
**1. Clone the Repository**
```bash
git clone https://github.com/devinazhifa/spam-detector.git
cd spam-detector
```

**2. Set Up the Environment**
Create a .env file in the project directory and add your Replicate API key:
```bash
REPLICATE_API_TOKEN=your_replicate_api_key
```

**3. Start the Web Interface**
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.
