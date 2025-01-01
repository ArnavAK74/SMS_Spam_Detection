from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from preprocess_text import preprocess_text

# Load the saved model
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="my_model.tflite")
interpreter.allocate_tensors()

# Load the saved tokenizer
import json
from keras.preprocessing.text import tokenizer_from_json
with open("tokenizer.json", "r") as f:
    tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data) # Ensure the tokenizer was saved during training

# Initialize the FastAPI app
app = FastAPI()

# Define the input schema for API
class SpamInput(BaseModel):
    text: str

# Define the prediction endpoint
@app.post("/predict")
async def predict_spam(input_data: SpamInput):
    # Step 1: Preprocess the input text
    processed_text = preprocess_text(input_data.text)
    print(f"Processed Text: {processed_text}")  # Debugging step

    # Step 2: Tokenize the input
    tokenized_input = tokenizer.texts_to_sequences([processed_text])
    print(f"Tokenized Input: {tokenized_input}")  # Debugging step

    # Step 3: Pad the tokenized sequence
    padded_input = pad_sequences(tokenized_input, maxlen=100)
    print(f"Padded Input: {padded_input}")  # Debugging step

    # Step 4: Predict using the model
    prediction = model.predict(padded_input)
    print(f"Prediction: {prediction}")  # Debugging step

    # Step 5: Interpret the prediction
    result = "Spam" if prediction[0][0] > 0.5 else "Not Spam"

    # Return the prediction result
    return {"text": input_data.text, "prediction": result}
