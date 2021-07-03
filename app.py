import streamlit as st
import joblib

from src.fingerprinter import get_fingerprints

st.title("Predict activity")

def load_model():
    mdl = joblib.load("model/my_first_model.pkl")
    return mdl

mdl = load_model()

smiles = st.text_input("Input molecules in SMILES format")

X = get_fingerprints([smiles])

y = mdl.predict(X)

st.text("My prediction is: {0}".format(y))
