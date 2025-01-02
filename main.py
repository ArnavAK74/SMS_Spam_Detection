from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import joblib
import numpy as np
import pickle
import os
import uvicorn
from preprocess_text import preprocess_text  # Ensure this file exists in your project

# Cloud storage URLs for model and tokenizer
MODEL_URL = "https://sms-spam-requirements.s3.us-east-2.amazonaws.com/my_model.tflite"
TOKENIZER_URL = "https://sms-spam-requirements.s3.us-east-2.amazonaws.com/tokenizer.pkl"

# Download and load the TensorFlow Lite model
def load_model(url, destination="my_model.tflite"):
    print("Downloading TensorFlow Lite model...")
    response = requests.get(url, stream=True)
    with open(destination, "wb") as f:
        f.write(response.content)
    print(f"Model downloaded and saved to {destination}")
    interpreter = tf.lite.Interpreter(model_path=destination)
    interpreter.allocate_tensors()
    return interpreter

# Download and load the tokenizer (pickle format)

# Function to load the tokenizer
from joblib import load
import requests

# Function to download and load the tokenizer
def load_tokenizer(url, destination="tokenizer.pkl"):
    print("Downloading Tokenizer...")
    response = requests.get(url, stream=True)
    with open(destination, "wb") as f:
        f.write(response.content)
    print(f"Tokenizer downloaded and saved to {destination}")
    
    # Use joblib to load the tokenizer
    tokenizer = load(destination)
    return tokenizer


# Load the Tokenizer

interpreter = load_model(MODEL_URL)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
tokenizer = load_tokenizer(TOKENIZER_URL)


# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)


# Define input schema
class SpamInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_spam(input_data: SpamInput):
    try:
        # Step 1: Preprocess the input text
        processed_text = preprocess_text(input_data.text)

        # Step 2: Tokenize and pad the input
        tokenized_input = tokenizer.texts_to_sequences([processed_text])
        padded_input = pad_sequences(tokenized_input, maxlen=100)

        # Step 3: Run inference using the TensorFlow Lite model
        interpreter.set_tensor(input_details[0]['index'], padded_input.astype('float32'))
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        # Step 4: Convert prediction to a Python float
        prediction_value = float(prediction)
        print(f"Prediction Score: {prediction_value}")

        # Step 5: Interpret the prediction
        result = "Spam" if prediction_value > 0.5 else "Not Spam"

        return {
            "text": input_data.text,
            "prediction": result,
            "score": prediction_value
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}
