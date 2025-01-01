import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define the text preprocessing function
def preprocess_text(text):
    """
    Preprocesses input text by:
    - Removing non-alphanumeric characters.
    - Removing digits.
    - Converting to lowercase.
    - Tokenizing and removing stopwords.
    - Lemmatizing words.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    # Remove non-alphanumeric characters
    text = re.sub(r'\W', ' ', text)
    # Remove digits
    text = re.sub(r'\d', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and clean
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)
