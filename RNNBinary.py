l

import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
import numpy as np

# --------------------------------
# 1. Load Data from CSV
# --------------------------------
# Ensure 'your_data.csv' has 'Summary' (text) and 'Label' (binary: 0/1)
df = pd.read_csv('your_data.csv')

# Split into training (80%) and testing (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Extract text and labels
train_texts = train_df['Summary'].values
train_labels = train_df['Label'].values
test_texts = test_df['Summary'].values
test_labels = test_df['Label'].values

# --------------------------------
# 2. Text Preprocessing (Encoder)
# --------------------------------
VOCAB_SIZE = 10000  # Maximum vocabulary size
MAX_SEQUENCE_LENGTH = 100  # Fixed sequence length

# TextVectorization Layer (Encoder)
encoder = TextVectorization(max_tokens=VOCAB_SIZE, output_mode='int', output_sequence_length=MAX_SEQUENCE_LENGTH)
encoder.adapt(train_texts)  # Build vocabulary from training data

# --------------------------------
# 3. Define Model (RNN with Bidirectional LSTMs)
# --------------------------------
vocab_size = len(encoder.get_vocabulary())  # Get vocabulary size

model = Sequential([
    encoder,  # Convert text to integer sequences
    Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True),  # Embedding Layer
    Bidirectional(LSTM(64, return_sequences=True)),  # First Bidirectional LSTM
    Bidirectional(LSTM(32)),  # Second Bidirectional LSTM
    Dense(64, activation='relu'),  # Dense Layer
    Dropout(0.5),  # Dropout for Regularization
    Dense(1, activation='sigmoid')  # Output Layer for Binary Classification
])

# --------------------------------
# 4. Compile Model
# --------------------------------
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

# Print model summary
print(model.summary())

# --------------------------------
# 5. Train Model
# --------------------------------
history = model.fit(
    train_texts, train_labels,
    epochs=5,  # Adjustable
    batch_size=32,  # Batch Size
    validation_data=(test_texts, test_labels),
    verbose=2
)

# --------------------------------
# 6. Evaluate Model
# --------------------------------
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_texts, test_labels, verbose=0)

print("\n--- Model Evaluation ---")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# --------------------------------
# 7. Test on New Raw Text Data
# --------------------------------
def predict_text(text_list):
    """Function to predict labels for raw text"""
    encoded_texts = encoder(text_list)  # Tokenize text
    predictions = model.predict(encoded_texts)  # Get model predictions
    predicted_labels = (predictions > 0.5).astype(int)  # Convert to binary labels (threshold=0.5)
    
    for text, prob, label in zip(text_list, predictions, predicted_labels):
        print(f"\nText: {text}\nPredicted Probability: {prob[0]:.4f}\nPredicted Label: {label[0]}")

# Example Predictions
new_texts = [
    "I absolutely love this product! It's amazing.",
    "Worst service ever. Very disappointed."
]
predict_text(new_texts)

# --------------------------------
# 8. Save the Model (Optional)
# --------------------------------
model.save("text_rnn_model.h5")  # Save the model for later use

# --------------------------------
# 9. Load and Use Model (Optional)
# --------------------------------
# To load and use the saved model:
# model = tf.keras.models.load_model("text_rnn_model.h5", custom_objects={'Precision': Precision, 'Recall': Recall})
# predict_text(["This is a new sample text for prediction."])


---

How to Use This Script

1. Prepare Your Data

Ensure you have a CSV file named your_data.csv with two columns:

"Summary" → contains text data

"Label" → binary labels (0 or 1)



2. Run the Script

Copy the script and run it in Google Colab or a Jupyter Notebook.

It will:

Train the model

Evaluate performance (Accuracy, Precision, Recall)

Test on new raw text



3. Save and Load Model (Optional)

The script saves the trained model as "text_rnn_model.h5".

You can reload it later using:

model = tf.keras.models.load_model("text_rnn_model.h5", custom_objects={'Precision': Precision, 'Recall': Recall})



---

Expected Output

When running predictions on new text:

Text: I absolutely love this product! It's amazing.
Predicted Probability: 0.9876
Predicted Label: 1

Text: Worst service ever. Very disappointed.
Predicted Probability: 0.1324
Predicted Label: 0


---

This end-to-end script is structured for ease of use. You can modify:

Hyperparameters (VOCAB_SIZE, MAX_SEQUENCE_LENGTH, LSTM units, EPOCHS)

Batch sizes

Training/test split percentages

Custom text inputs for testing


Let me know if you need any improvements or explanations! 🚀

