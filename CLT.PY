# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text as text
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Bidirectional, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Sample dataset: Replace with actual dataset
data = {
    'description': [
        "Server crash when high traffic",
        "Database timeout error",
        "Login page slow loading",
        "Memory leak issue in service",
        "App crashes on startup",
        "Slow response from API",
        "Authentication failure error",
        "Application not responding",
        "User unable to reset password",
        "Performance degraded during peak hours"
    ],
    'label': [0, 1, 2, 3, 0, 2, 1, 0, 1, 3]  # Example labels (0, 1, 2, 3)
}

df = pd.DataFrame(data)

# Splitting data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['description'], df['label'], test_size=0.2, random_state=42)

# Tokenizer settings
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 50  # Adjust based on text length

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

train_sequences = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_SEQUENCE_LENGTH, padding="post")
test_sequences = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=MAX_SEQUENCE_LENGTH, padding="post")

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Number of classes
NUM_CLASSES = len(set(train_labels))

# Define CNN Model
def build_cnn_model():
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, 128, input_length=MAX_SEQUENCE_LENGTH),
        Conv1D(128, 5, activation="relu"),
        MaxPooling1D(2),
        Conv1D(64, 3, activation="relu"),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
    return model

cnn_model = build_cnn_model()

# Define LSTM Model
def build_lstm_model():
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, 128, input_length=MAX_SEQUENCE_LENGTH),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
    return model

lstm_model = build_lstm_model()

# Transformer-based model (BERT)
bert_model_name = "distilbert-base-uncased"  # Lightweight BERT model
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

def encode_texts(texts):
    return bert_tokenizer(list(texts), padding=True, truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="tf")

train_encodings = encode_texts(train_texts)
test_encodings = encode_texts(test_texts)

bert_model = TFAutoModel.from_pretrained(bert_model_name)

def build_transformer_model():
    input_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="attention_mask")
    
    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(bert_output)
    
    dense = tf.keras.layers.Dense(128, activation="relu")(pooled_output)
    dropout = tf.keras.layers.Dropout(0.5)(dense)
    output = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(dropout)
    
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=2e-5), metrics=["accuracy"])
    
    return model

transformer_model = build_transformer_model()

# Training parameters
EPOCHS = 10
BATCH_SIZE = 16
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# Train CNN Model
cnn_model.fit(train_sequences, train_labels, validation_data=(test_sequences, test_labels), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping])

# Train LSTM Model
lstm_model.fit(train_sequences, train_labels, validation_data=(test_sequences, test_labels), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping])

# Train Transformer Model
transformer_model.fit(
    {"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"]},
    train_labels,
    validation_data=(
        {"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"]}, 
        test_labels
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping]
)

# Function to evaluate models
def evaluate_model(model, test_x, test_y):
    predictions = np.argmax(model.predict(test_x), axis=1)
    print(classification_report(test_y, predictions))
    
    cm = confusion_matrix(test_y, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Evaluate models
print("CNN Model Performance:")
evaluate_model(cnn_model, test_sequences, test_labels)

print("LSTM Model Performance:")
evaluate_model(lstm_model, test_sequences, test_labels)

print("Transformer Model Performance:")
evaluate_model(transformer_model, {"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"]}, test_labels)
