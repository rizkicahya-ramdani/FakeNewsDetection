import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing.text_cleaning import clean_text

# Load LIAR dataset
def load_data():
    cols = ['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job_title', 
            'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 
            'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']
    train = pd.read_csv('data/train.tsv', sep='\t', header=None, names=cols)
    test = pd.read_csv('data/test.tsv', sep='\t', header=None, names=cols)
    valid = pd.read_csv('data/valid.tsv', sep='\t', header=None, names=cols)
    df = pd.concat([train, test, valid])
    return df

df = load_data()

# Gunakan hanya kolom 'statement' dan 'label'
df = df[['statement', 'label']]
df.dropna(inplace=True)

# Bersihkan teks
df['clean_text'] = df['statement'].apply(clean_text)

# Encode label jadi biner: hoax (false, pants-fire, barely-true) vs non-hoax (true, mostly-true, half-true)
hoax_labels = ['false', 'pants-fire', 'barely-true']
df['label_binary'] = df['label'].apply(lambda x: 1 if x in hoax_labels else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label_binary'], test_size=0.2, random_state=42, stratify=df['label_binary']
)

# Tokenization & Padding
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Build Model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=64),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test))

# Simpan model dan tokenizer
os.makedirs('model', exist_ok=True)
model.save('model/lstm_model.h5')
with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("✅ Model and tokenizer saved.")
