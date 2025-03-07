import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Sample Data (Replace with Your Dataset) ---
problems = ["My internet connection is very slow.", "I can't log into my account."]
resolutions = ["Try restarting your router or contact your service provider.", "Reset your password or check if the caps lock is on."]

# Add special tokens to the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(problems + resolutions + ["<start>", "<end>"])
vocab_size = len(tokenizer.word_index) + 1  # +1 for the 0 index

# Convert text to sequences
problem_sequences = tokenizer.texts_to_sequences(problems)
resolution_sequences = tokenizer.texts_to_sequences(resolutions)

# Determine maximum lengths
max_problem_length = max(len(seq) for seq in problem_sequences)
max_resolution_length = max(len(seq) for seq in resolution_sequences)

# Pad sequences
problem_sequences = pad_sequences(problem_sequences, maxlen=max_problem_length, padding='post')
resolution_sequences = pad_sequences(resolution_sequences, maxlen=max_resolution_length, padding='post')

# Prepare decoder input and target sequences
decoder_input_sequences = [[tokenizer.word_index["<start>"]] + seq for seq in resolution_sequences]
decoder_target_sequences = [seq + [tokenizer.word_index["<end>"]] for seq in resolution_sequences]
decoder_input_sequences = pad_sequences(decoder_input_sequences, maxlen=max_resolution_length + 1, padding='post')
decoder_target_sequences = pad_sequences(decoder_target_sequences, maxlen=max_resolution_length + 1, padding='post')

# --- Model Parameters ---
embedding_dim = 256  # Size of word embeddings
lstm_units = 512     # Number of LSTM units

# --- Encoder ---
encoder_inputs = Input(shape=(max_problem_length,))
encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(lstm_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# --- Decoder ---
decoder_inputs = Input(shape=(max_resolution_length + 1,))
decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# --- Training Model ---
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# --- Train the Model (Uncomment and adjust epochs/batch_size with real data) ---
# model.fit([problem_sequences, decoder_input_sequences], decoder_target_sequences, epochs=10, batch_size=32)

# --- Inference Setup ---
# Encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model
decoder_state_input_h = Input(shape=(lstm_units,))
decoder_state_input_c = Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# --- Function to Generate Resolution ---
def generate_resolution(problem):
    # Tokenize and pad the input problem
    problem_seq = tokenizer.texts_to_sequences([problem])
    problem_seq = pad_sequences(problem_seq, maxlen=max_problem_length, padding='post')
    
    # Encode the problem
    states_value = encoder_model.predict(problem_seq, verbose=0)
    
    # Initialize decoder input with <start> token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index["<start>"]
    
    # Generate resolution
    resolution = []
    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, "")
        
        if sampled_word == "<end>" or len(resolution) >= max_resolution_length:
            break
        resolution.append(sampled_word)
        
        # Update target sequence and states
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    
    return " ".join(resolution)

# --- Example Usage ---
new_problem = "My internet connection is very slow."
resolution = generate_resolution(new_problem)
print(f"Problem: {new_problem}")
print(f"Resolution: {resolution}")
