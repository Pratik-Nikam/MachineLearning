
stop_words = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'i', 'you', 'they', 'we', 'this', 'but', 'or', 'not'
}

# Keep domain-relevant words
stop_words -= {''} 

def remove_stop_words(text, stop_words):
    """
    Remove stop words from a text string.
    
    Args:
        text (str): Input text.
        stop_words (set): Set of stop words to remove.
    
    Returns:
        str: Text with stop words removed.
    """
    # Split text into words
    words = text.split()
    # Keep words that aren't stop words
    filtered_words = [word for word in words if word not in stop_words]
    # Join words back into a string
    return ' '.join(filtered_words)


import pandas as pd


df['summary'] = df['summary'].str.lower()


df['summary'] = df['summary'].apply(lambda x: remove_stop_words(x, stop_words))



import spacy

nlp = spacy.blank('en')  # Blank English pipeline
nlp.add_pipe('lemmatizer', config={'mode': 'rule'})  # Rule-based lemmatizer

def lemmatize_text(text):
    """
    Lemmatize a text string using spacy's rule-based lemmatizer.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Lemmatized text.
    """
    doc = nlp(text)
    return ' '.join(token.lemma_ for token in doc)

df['descriptions'] = df['descriptions'].apply(lemmatize_text)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame with cleaned 'descriptions' and 'labels'
# Assuming vectorizer and label_encoder are already defined and fitted

# Step 1: Tokenize and split data (already done, but repeated for clarity)
max_tokens = 5000
max_len = 50

vectorizer = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_len,
    ngrams=2
)
vectorizer.adapt(df['descriptions'].values)

X = vectorizer(np.array(df['descriptions'])).numpy()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['labels'].values)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Step 2: Create a mapping from token IDs back to words
# Get vocabulary from TextVectorization layer
vocab = vectorizer.get_vocabulary()  # List of tokens (index 0 is '', 1 is '[UNK]', etc.)
id_to_word = {i: word for i, word in enumerate(vocab)}

# Function to decode tokenized sequences back to text
def decode_sequence(tokenized_seq):
    """
    Decode a tokenized sequence back to text.
    
    Args:
        tokenized_seq (np.ndarray): Array of token IDs (shape: (max_len,)).
    
    Returns:
        str: Decoded text.
    """
    # Convert token IDs to words, ignoring padding (0) and [UNK] (1) tokens
    words = [id_to_word.get(token, '') for token in tokenized_seq if token not in [0, 1]]
    return ' '.join(words)

# Step 3: Function to print samples from a dataset split
def print_samples(X_data, y_data, split_name, num_samples=5):
    """
    Print a few samples from a dataset split.
    
    Args:
        X_data (np.ndarray): Tokenized sequences.
        y_data (np.ndarray): Encoded labels.
        split_name (str): Name of the split (e.g., 'Training').
        num_samples (int): Number of samples to print.
    """
    print(f"\n--- {split_name} Data Samples ---")
    # Randomly select indices to sample
    indices = np.random.choice(len(X_data), num_samples, replace=False)
    
    for idx in indices:
        # Decode the tokenized sequence back to text
        text = decode_sequence(X_data[idx])
        # Decode the label back to text
        label = label_encoder.inverse_transform([y_data[idx]])[0]
        print(f"Description: {text}")
        print(f"Label: {label}")
        print()

# Step 4: Print samples from each split
print_samples(X_train, y_train, "Training", num_samples=5)
print_samples(X_val, y_val, "Validation", num_samples=5)
print_samples(X_test, y_test, "Testing", num_samples=5)
