
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



import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re

# Load your data (replace with your actual data loading)
# For demonstration, create dummy data
np.random.seed(42)
data = {
    'descriptions': ['sample text ' * np.random.randint(10, 50) for _ in range(10000)],
    'labels': np.random.choice(['tech issue', 'other', 'end user training', 'bypass'], size=10000)
}
df = pd.DataFrame(data)

# Step 1: Convert 4 labels to 2 labels
# Example grouping: 
# - "tech issue" and "end user training" -> "technical/support" (label 1)
# - "other" and "bypass" -> "non-technical" (label 0)
def convert_to_binary_labels(label):
    if label in ['tech issue', 'end user training']:
        return 'technical/support'
    else:
        return 'non-technical'

df['labels'] = df['labels'].apply(convert_to_binary_labels)

# Step 2: Clean descriptions
def clean_text(text):
    text = text.lower()
    # Remove boilerplate phrases
    boilerplate_phrases = [
        'please contact support',
        'kindly see the attached screen shot',
        'processing completed with error',
        'unable to submit approve'
    ]
    for phrase in boilerplate_phrases:
        text = text.replace(phrase, ' ')
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join(text.split())
    return text

df['descriptions'] = df['descriptions'].apply(clean_text)

# Step 3: Encode labels (binary: 0 and 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['labels'])  # "non-technical" -> 0, "technical/support" -> 1

# Step 4: Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    df['descriptions'].values, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Step 5: Preprocess text data with TextVectorization
max_tokens = 10000  # Vocabulary size
max_length = 50     # Sequence length (based on your data distribution: most <40 words)
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=max_length
)
vectorize_layer.adapt(X_train)

# Step 6: Build the RNN model (adapted from TensorFlow tutorial)
model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Embedding(max_tokens + 1, 128),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.4),  # Prevent overfitting
    tf.keras.layers.Dense(64, activation='relu', 
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
model.summary()

# Step 7: Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    mode='max',
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    mode='max'
)

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Step 8: Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Step 9: Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 10: Save the model for retraining
model.save('rnn_binary_classifier_initial.tf', save_format='tf')

# Step 11: Retrain the model (example with lower learning rate)
# Load the model
model = tf.keras.models.load_model('rnn_binary_classifier_initial.tf')

# Adjust settings for retraining
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Retrain for a few more epochs
history_retrain = model.fit(
    X_train, y_train,
    epochs=5,  # Additional epochs
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Step 12: Evaluate again after retraining
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy after Retraining: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Step 13: Prediction on new data
def predict_rnn(texts, model, label_encoder):
    """
    Predict labels for new texts using the RNN model.
    
    Args:
        texts (list or str): Text(s) to predict.
        model: Trained RNN model.
        label_encoder: LabelEncoder for decoding labels.
    
    Returns:
        list: Predicted labels.
    """
    if isinstance(texts, str):
        texts = [texts]
    
    # Preprocess texts (clean and vectorize automatically via model)
    predictions = model.predict(texts, verbose=0)
    predicted_indices = (predictions > 0.5).astype(int).flatten()
    predicted_labels = label_encoder.inverse_transform(predicted_indices)
    
    return predicted_labels.tolist()

# Example prediction
new_texts = [
    "tech issue encountered when adding related party",
    "end user training required for new staff",
    "bypass this step for now",
    "generic issue unrelated"
]
predictions = predict_rnn(new_texts, model, label_encoder)
print("Predictions:", predictions)

# Step 14: Print a few samples from each split
def print_samples(X_data, y_data, split_name, num_samples=5):
    """
    Print a few samples from a dataset split.
    
    Args:
        X_data (np.ndarray): Descriptions.
        y_data (np.ndarray): Encoded labels.
        split_name (str): Name of the split (e.g., 'Training').
        num_samples (int): Number of samples to print.
    """
    print(f"\n--- {split_name} Data Samples ---")
    sample_indices = np.random.choice(len(X_data), num_samples, replace=False)
    
    for idx in sample_indices:
        text = X_data[idx]
        label = label_encoder.inverse_transform([y_data[idx]])[0]
        print(f"Description: {text}")
        print(f"Label: {label}")
        print()

# Print samples
print_samples(X_train, y_train, "Training")
print_samples(X_val, y_val, "Validation")
print_samples(X_test, y_test, "Testing")




