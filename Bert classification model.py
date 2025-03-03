import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Load your data (replace with your actual data loading)
# For demonstration, create dummy data
np.random.seed(42)
data = {
    'descriptions': ['sample text ' * np.random.randint(10, 50) for _ in range(10000)],
    'labels': np.random.choice(['t', 'o', 'e', 'b'], size=10000)
}
df = pd.DataFrame(data)

# Step 1: Load BERT model and tokenizer
bert_model_name = "bert-base-uncased"  # Replace with your BERT model name if different
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = TFBertModel.from_pretrained(bert_model_name)

# If you have custom BERT weights, load them here (uncomment and adjust path)
# bert_model.load_weights('path/to/your/bert_weights')

# Step 2: Preprocess data for BERT
def encode_texts(texts, tokenizer, max_length=100):
    """
    Encode texts using BERT tokenizer.
    
    Args:
        texts (list): List of text descriptions.
        tokenizer: BERT tokenizer.
        max_length (int): Maximum sequence length.
    
    Returns:
        dict: Dictionary with input_ids, attention_mask, and token_type_ids.
    """
    encoded = tokenizer(
        texts.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf',
        return_attention_mask=True,
        return_token_type_ids=True
    )
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'token_type_ids': encoded['token_type_ids']
    }

# Encode descriptions
encoded_data = encode_texts(df['descriptions'], tokenizer, max_length=100)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['labels'].values)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    np.arange(len(df)), y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Create TensorFlow datasets
def create_dataset(encoded_data, indices, labels):
    return tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': tf.gather(encoded_data['input_ids'], indices),
            'attention_mask': tf.gather(encoded_data['attention_mask'], indices),
            'token_type_ids': tf.gather(encoded_data['token_type_ids'], indices)
        },
        labels
    )).shuffle(1000).batch(16)

train_dataset = create_dataset(encoded_data, X_train, y_train)
val_dataset = create_dataset(encoded_data, X_val, y_val)
test_dataset = create_dataset(encoded_data, X_test, y_test)

# Step 3: Build BERT-based model
def build_bert_model(bert_model, num_labels, max_length=100):
    """
    Build a BERT-based classification model.
    
    Args:
        bert_model: Pre-trained BERT model (TFBertModel).
        num_labels (int): Number of output labels.
        max_length (int): Maximum sequence length.
    
    Returns:
        tf.keras.Model: Compiled model.
    """
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')
    token_type_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='token_type_ids')
    
    bert_outputs = bert_model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )
    cls_output = bert_outputs.pooler_output
    
    x = tf.keras.layers.Dropout(0.1)(cls_output)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    output = tf.keras.layers.Dense(num_labels, activation="softmax")(x)
    
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=output
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Build model
num_labels = len(label_encoder.classes_)
bert_classifier = build_bert_model(bert_model, num_labels)
bert_classifier.summary()

# Step 4: Train the model
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

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

history = bert_classifier.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights_dict,
    verbose=1
)

# Evaluate on test set
test_loss, test_accuracy = bert_classifier.evaluate(test_dataset, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Plot training history
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

# Step 5: Prediction on new data
def predict_bert(texts, model, tokenizer, label_encoder, max_length=100):
    """
    Predict labels for new texts using the BERT model.
    
    Args:
        texts (list or str): Text(s) to predict.
        model: Trained BERT model.
        tokenizer: BERT tokenizer.
        label_encoder: LabelEncoder for decoding labels.
        max_length (int): Maximum sequence length.
    
    Returns:
        list: Predicted labels.
    """
    if isinstance(texts, str):
        texts = [texts]
    
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf',
        return_attention_mask=True,
        return_token_type_ids=True
    )
    
    predictions = model.predict(
        [encoded['input_ids'], encoded['attention_mask'], encoded['token_type_ids']],
        verbose=0
    )
    
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_indices)
    
    return predicted_labels.tolist()

# Example prediction
new_texts = [
   
]
predictions = predict_bert(new_texts, bert_classifier, tokenizer, label_encoder)
print("Predictions:", predictions)

# Step 6: Print a few samples from each split
def print_samples(indices, texts, labels, split_name, num_samples=5):
    """
    Print a few samples from a dataset split.
    
    Args:
        indices (np.ndarray): Indices of the split.
        texts (pd.Series): Original descriptions.
        labels (np.ndarray): Encoded labels.
        split_name (str): Name of the split (e.g., 'Training').
        num_samples (int): Number of samples to print.
    """
    print(f"\n--- {split_name} Data Samples ---")
    sample_indices = np.random.choice(indices, num_samples, replace=False)
    
    for idx in sample_indices:
        text = texts.iloc[idx]
        label = label_encoder.inverse_transform([labels[idx]])[0]
        print(f"Description: {text}")
        print(f"Label: {label}")
        print()

# Print samples
print_samples(X_train, df['descriptions'], y_train, "Training")
print_samples(X_val, df['descriptions'], y_val, "Validation")
print_samples(X_test, df['descriptions'], y_test, "Testing")
