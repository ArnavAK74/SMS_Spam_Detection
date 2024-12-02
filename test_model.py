from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the saved model
model = load_model("my_model.h5")

# Load or recreate the tokenizer
tokenizer = Tokenizer(num_words=3000)  # Ensure the parameters match what you used in training

# Example: Refit tokenizer if you saved training data or tokenizer separately
# tokenizer.fit_on_texts(data['processed_message'])  # Only needed if training data is available

# Input text for prediction
sample_input = [" You've won a freeasdad sare sare jahanhg se acha."]

# Preprocess the input
tokenized_input = tokenizer.texts_to_sequences(sample_input)  # Tokenize
padded_input = pad_sequences(tokenized_input, maxlen=100)  # Pad to match input shape

# Make predictions
prediction = model.predict(padded_input)
print(f"Prediction: {prediction}")

# Interpret the result (assuming binary classification)
predicted_label = "Spam" if prediction[0] > 0.5 else "Not Spam"
print(f"Predicted Label: {predicted_label}")


