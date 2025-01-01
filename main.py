from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from preprocess_text import preprocess_text  # Ensure this file exists in your project

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="my_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the saved tokenizer
tokenizer = joblib.load("tokenizer.pkl")  # Ensure tokenizer.pkl is in your project directory

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running. Use /predict to test predictions."}

# Define input schema
class SpamInput(BaseModel):
    text: str

# API endpoint
@app.post("/predict")
async def predict_spam(input_data: SpamInput):
    # Step 1: Preprocess the input text
    processed_text = preprocess_text(input_data.text)

    # Step 2: Tokenize and pad the input
    try:
        tokenized_input = tokenizer.texts_to_sequences([processed_text])
    except AttributeError:
        return {"error": "Tokenizer is not loaded or not compatible. Please check tokenizer.pkl."}
    
    if not tokenized_input or len(tokenized_input[0]) == 0:
        return {"error": "Input text could not be tokenized. Please check your input or tokenizer."}

    padded_input = pad_sequences(tokenized_input, maxlen=100)

    # Step 3: Run inference using the TensorFlow Lite model
    interpreter.set_tensor(input_details[0]['index'], padded_input.astype('float32'))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    print(f"Prediction probability: {prediction}")

    # Step 4: Interpret the prediction
    result = "Spam" if prediction > 0.5 else "Not Spam"

    return {"text": input_data.text, "prediction": result}
