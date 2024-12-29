import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import string
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical


file_path = 'cleaned_output.csv'
data = pd.read_csv(file_path, sep=';', engine='python')



if 'URL' not in data.columns or 'classification' not in data.columns:
    raise ValueError("Data must have 'URL' and 'classification' columns.")

urls = data['URL'].astype(str)
labels = data['classification'].astype(str)


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

allowed_chars = string.printable  
allowed_chars = allowed_chars.replace('\r', '').replace('\n', '').replace('\t', '')

char_to_idx = {c: i+1 for i, c in enumerate(allowed_chars)}  
idx_to_char = {i: c for c, i in char_to_idx.items()}


def url_to_seq(url, char_to_idx):
    return [char_to_idx.get(c, 0) for c in url]  

X_seq = [url_to_seq(u, char_to_idx) for u in urls]


max_length = 200  
X_pad = pad_sequences(X_seq, maxlen=max_length, padding='post', truncating='post')

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.3, random_state=42)

# -----------------------------
# Build the Model
# -----------------------------
vocab_size = len(char_to_idx) + 1  # +1 for padding
embedding_dim = 64
lstm_units = 64

model = Sequential()
model.add(Embedding(input_dim=vocab_size, 
                    output_dim=embedding_dim, 
                    input_length=max_length))
model.add(LSTM(lstm_units))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# -----------------------------
# Train the Model
# -----------------------------
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# -----------------------------
# Evaluate the Model
# -----------------------------
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.2f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# -----------------------------
# Save the Model and Encoders
# -----------------------------
model.save('char_lstm_model.h5')
with open('char_to_idx.pkl', 'wb') as f:
    pickle.dump(char_to_idx, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
