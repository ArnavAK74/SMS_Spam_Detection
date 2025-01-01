from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib  # For loading the trained tokenizer
from preprocess_text import preprocess_text  # Import the preprocessing function

# Load the saved model
model = load_model("my_model.h5")

# Load the saved tokenizer
tokenizer = joblib.load("tokenizer.pkl")  # Load the tokenizer saved during training

# Example input
sample_text = "Congratulations! You've won $10,000 by clicking this link."

# Step 1: Preprocess the input
processed = preprocess_text(sample_text)  # Preprocess the input text
print(f"Processed Text: {processed}")

# Step 2: Tokenize the preprocessed input
tokenized_input = tokenizer.texts_to_sequences([processed])  # Tokenize as a list
print(f"Tokenized Input: {tokenized_input}")

# Step 3: Pad the tokenized input
padded_input = pad_sequences(tokenized_input, maxlen=100)  # Pad sequences to match model input shape
print(f"Padded Input: {padded_input}")

# Step 4: Make predictions
prediction = model.predict(padded_input)
print(f"Prediction: {prediction}")  # Model's probability output

# Step 5: Interpret the result
predicted_label = "Spam" if prediction[0][0] > 0.5 else "Not Spam"
print(f"Predicted Label: {predicted_label}")
