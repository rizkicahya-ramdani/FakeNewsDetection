import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing.text_cleaning import clean_text

# Load model & tokenizer
model = load_model('model/lstm_model.h5')
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 100

# UI
st.title("🕵️ Fake News Detection (English)")
st.write("Enter a political statement to detect whether it is likely **Hoax** or **Non-Hoax**.")

user_input = st.text_area("📝 Statement")

if st.button("Detect"):
    cleaned = clean_text(user_input)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded)[0][0]
    
    label = "HOAX" if prediction >= 0.5 else "NON-HOAX"
    st.subheader("📊 Detection Result:")
    st.write(f"**Label:** {label}")
    st.write(f"**Probability:** {prediction:.4f}")
