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


