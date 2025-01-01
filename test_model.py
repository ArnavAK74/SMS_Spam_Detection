import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib  # For loading the trained tokenizer
from preprocess_text import preprocess_text  # Import the preprocessing function

# Step 1: Load the TensorFlow Lite model
try:
    interpreter = tf.lite.Interpreter(model_path="my_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded successfully.")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    exit()

# Step 2: Load the tokenizer
try:
    tokenizer = joblib.load("tokenizer.pkl")
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# Example input
sample_text = "Congratulations! You've won free free free and $10,000 by clicking this link."

# Step 3: Preprocess the input
try:
    processed = preprocess_text(sample_text)  # Preprocess the input text
    print(f"Processed Text: {processed}")
except Exception as e:
    print(f"Error during text preprocessing: {e}")
    exit()

# Step 4: Tokenize the preprocessed input
try:
    tokenized_input = tokenizer.texts_to_sequences([processed])  # Tokenize as a list
    print(f"Tokenized Input: {tokenized_input}")
except Exception as e:
    print(f"Error during tokenization: {e}")
    exit()

# Step 5: Pad the tokenized input
try:
    padded_input = pad_sequences(tokenized_input, maxlen=100)  # Pad sequences to match model input shape
    print(f"Padded Input: {padded_input}")
except Exception as e:
    print(f"Error during padding: {e}")
    exit()

# Step 6: Prepare the input for the TFLite model
try:
    input_data = tf.convert_to_tensor(padded_input, dtype=input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    print(f"Prediction: {prediction}")
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()

# Step 7: Interpret the result
try:
    predicted_label = "Spam" if prediction[0][0] > 0.5 else "Not Spam"
    print(f"Predicted Label: {predicted_label}")
except Exception as e:
    print(f"Error during result interpretation: {e}")
    exit()
