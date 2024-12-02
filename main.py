from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the saved model
model = load_model("my_model.h5")



# Initialize the FastAPI app
app = FastAPI()

# Define the input schema
class SpamInput(BaseModel):
    text: str

# Define the prediction endpoint
@app.post("/predict")
async def predict_spam(input_data: SpamInput):
    # Predict spam or not
    tokenizer = Tokenizer(num_words=3000) 

    tokenized_input = tokenizer.texts_to_sequences([input_data.text])  # Tokenize
    padded_input = pad_sequences(tokenized_input, maxlen=100)
    prediction = model.predict(padded_input)
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return {"text": input_data.text, "prediction": result}
