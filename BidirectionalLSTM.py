import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

texts = [
    "My laptop won't turn on",
    "How do I reset my password?",
    "The network is down again",
    "I need help with software installation",
] * 2500  # Dummy dataset (10,000 samples)

labels = [
    "tech issue", "bypass", "other", "end user training"
] * 2500  # Corresponding labels

# Encode text labels into numeric labels (0, 1, 2, 3)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)  # Converts text labels to numbers

# Split data into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(texts, labels_encoded, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

# Text Vectorization
max_words = 20000  # Max vocabulary size
seq_length = 200  # Max sequence length in tokens

vectorizer = layers.TextVectorization(max_tokens=max_words, output_sequence_length=seq_length, standardize="lower_and_strip_punctuation")
vectorizer.adapt(X_train)  # Learn vocabulary from training texts

# Model Architecture
embedding_dim = 128  # Embedding output dimension
vocab_size = len(vectorizer.get_vocabulary())  # Vocabulary size

model = models.Sequential([
    vectorizer,  # Convert text to token sequences
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
    layers.Bidirectional(layers.LSTM(64, dropout=0.3, return_sequences=False)),  # LSTM Layer
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # Dense Layer
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')  # Output layer for 4-class classification
])

# Compile the Model
model.compile(loss='sparse_categorical_crossentropy',  # Suitable for integer labels (0-3)
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=['accuracy'])

# Define Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)

# Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop, reduce_lr])

# Evaluate on Test Set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# Predict on New Data
new_texts = ["How do I change my WiFi password?", "My screen is not displaying anything"]
predictions = model.predict(new_texts)
predicted_classes = np.argmax(predictions, axis=1)  # Get the highest probability class

# Decode numeric predictions back to text labels
predicted_labels = label_encoder.inverse_transform(predicted_classes)
print(f"Predictions: {predicted_labels}")


# Fine Tuning 

def make_model():
    model = models.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(input_dim=max_tokens, output_dim=128),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(4, activation="softmax")
    ])
    return model

model = make_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# run 2

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Step 1: Data Preparation ---
# Replace this with your actual data loading
# For demonstration, I'll create dummy data with your labels
np.random.seed(42)
data = {
    'descriptions': ['sample text ' * np.random.randint(10, 50) for _ in range(10000)],
    'labels': np.random.choice(['tech issue', 'other', 'end user training', 'bypass'], size=10000)
}
df = pd.DataFrame(data)

# Convert text labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['labels'].values)

# Print label mapping for reference
label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print("Label Mapping:", label_mapping)
# Output: {'b': 0, 'e': 1, 'o': 2, 't': 3}

# Define preprocessing parameters
max_tokens = 10000  # Vocabulary size
max_len = 200       # Max sequence length (covers ~800 characters)

# Text vectorization layer
vectorizer = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_len
)

# Adapt the vectorizer to your text data
vectorizer.adapt(df['descriptions'].values)

# Convert text to sequences
X = vectorizer(np.array(df['descriptions'])).numpy()
y = y_encoded  # Use encoded integer labels

# Split into train, validation, and test sets (80/10/10)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# --- Step 2: Define the Model ---
def make_model():
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(max_len,)),
        # Embedding layer: converts tokens to 128-dimensional vectors
        layers.Embedding(input_dim=max_tokens, output_dim=128),
        # Bidirectional LSTM: captures forward and backward context
        layers.Bidirectional(layers.LSTM(64)),
        # Dense layers with dropout for classification
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),  # Regularization to prevent overfitting
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        # Output layer: 4 classes with softmax
        layers.Dense(4, activation="softmax")
    ])
    return model

# Create and compile the model
model = make_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # For integer labels
    metrics=["accuracy"]
)

# Model summary
model.summary()

# --- Step 3: Train the Model ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,  # Adjust based on convergence
    batch_size=32,
    verbose=1
)

# --- Step 4: Evaluate on Test Data ---
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# --- Step 5: Predict and Decode Labels (Optional) ---
# Example prediction on test data
predictions = model.predict(X_test[:5])
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
print("\nExample Predictions:", predicted_labels)

# --- Optional: Plot Training History ---
import matplotlib.pyplot as plt

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
