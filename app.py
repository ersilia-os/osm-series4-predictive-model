import streamlit as st
import joblib
from rdkit import Chem
from rdkit.Chem.Descriptors import qed
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Descriptors import MolLogP

from src.fingerprinter import get_fingerprints

st.title("Property prediction for OSM series 4 molecules")

def load_model(path):
    mdl = joblib.load(path)
    return mdl

smiles = st.text_input("Input molecules in SMILES format", value = "")

X = get_fingerprints([smiles])

st.header("Predicted Characteristics")
col1, col2, col3, col4, col5 =st.beta_columns(5) 

col1.subheader("Activity")
mdl = load_model("model/my_first_model.pkl")
y = mdl.predict(X)
col1.write("{0}".format(y), help="Results interpretation: 0 = no activity - 1 = maximum activity")

col2.subheader("MW")
mw=ExactMolWt(Chem.MolFromSmiles(smiles))
col2.write("{0:.2f}".format(mw))

col3.subheader("Solubility")
logp=MolLogP(Chem.MolFromSmiles(smiles))
col3.write("{0:.2f}".format(logp))

col4.subheader("QED")
qed_score=qed(Chem.MolFromSmiles(smiles))
col4.write("{0:.2f}".format(qed_score))




#activity
st.subheader("Activity prediction")
st.write("Activity against the malaria parasite is predicted using the ML model [eoschem](https://github.com/ersilia-os/ersilia-automl-chem). This model has been trained using the dataset provided by the [Open Source Malaria Project](https://github.com/opensourcemalaria) and only takes into account series 4 molecules. Thus, it will be most accurate for molecules from the same series.")
mdl = load_model("model/my_first_model.pkl")
y = mdl.predict(X)
st.write("Antimalarial Predicted Activity: {0}".format(y))
st.text("Results interpretation: 0 = no activity - 1 = maximum activity")

#QED
st.subheader("QED prediction")
st.write("Estimation of the drug likeness using the QED score as calculated by rdkit")
qed_score=qed(Chem.MolFromSmiles(smiles))
st.write("QED score: {0}".format(qed_score))

#Molecular weight
st.subheader("Molecular Weight")
mw=ExactMolWt(Chem.MolFromSmiles(smiles))
st.write("MW: {0}".format(mw))

#Solubility
st.subheader("Solubility")
st.write("Solubility prediction (LogP)")
logp=MolLogP(Chem.MolFromSmiles(smiles))
st.write("SLogP: {0}".format(logp))
st.text("Interpretation: values <3.5 are considered good")


"""
#RA_score
mdl = load_model("model/model_RA.pkl")
y = mdl.predict(X)
st.write("Retrosynthetic Accessibility Score: {0}".format(y))
st.text("Results interpretation: 0 = no activity - 1 = maximum activity")
"""