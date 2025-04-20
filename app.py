# app.py

import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("svm_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Spam Detector", page_icon="📧")
st.title("📩 Spam Mail Classifier")
st.write("Enter your email message below and check if it's spam or not.")

# User input
user_input = st.text_area("✉️ Type your message here")

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)

        if prediction[0] == 1:
            st.error("🚨 This message is likely SPAM.")
        else:
            st.success("✅ This message is likely HAM (not spam).")
