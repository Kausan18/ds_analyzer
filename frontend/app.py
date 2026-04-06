import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.title("Dataset Evaluator")

if st.button("Test Backend Connection"):
    try:
        response = requests.get(f"{BACKEND_URL}/ping")
        if response.status_code == 200:
            st.success(f"Connected! Response: {response.json()['message']}")
        else:
            st.error("Backend returned an error")
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach backend. Is it running?")