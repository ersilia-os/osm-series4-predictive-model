import streamlit as st
import joblib
import pandas as pd
import numpy as np
import onnxruntime as rt
import json
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Descriptors import MolLogP
from rdkit.Chem import Draw
from rdkit.DataStructs import BulkTanimotoSimilarity

from src.fingerprinter import get_fingerprints, mols_to_fingerprints, ra_fingerprint
from src.eosdescriptors.chembl import Chembl
from src.eosdescriptors.ecfp import Ecfp
from src.eosdescriptors.rdkit2d import Rdkit2d


# APP:

st.title("Properties calculation for OSM Series 4 molecules")

# First section: molecule input and drawing
col1, col2 = st.beta_columns(2)
#Input
col1.subheader("Input molecule in SMILES format")
smiles = col1.text_input("", value = "")
if not smiles:
    mol = None
else:
    mol=Chem.MolFromSmiles(smiles)
#Image
col2.subheader("Molecule display")
if mol is not None:
    col2.image(Draw.MolToImage(mol), width=200)

# Second section: prediction of properties
st.header("Properties")
col1, col2, col3, col4, col5, col6 =st.beta_columns(6)

col1.subheader("Activity")
MODELS_DIR = "model"

models_files = {}
with open(os.path.join(MODELS_DIR, "models.json"), "r") as f:
    all_models = json.load(f)
for dn, tn in all_models:
    models_files[(dn, tn)] = os.path.join(MODELS_DIR, "{0}_{1}.pkl".format(dn, tn))

sessions = {}
for k, v in models_files.items():
    sess = joblib.load(v)
    sessions[k] = sess

necessary_descriptors = sorted(set([k[0] for k in all_models]))
necessary_tasks = sorted(set([k[1] for k in all_models]))

descriptor_calculators = {
    "chembl": Chembl(),
    "ecfp": Ecfp(),
    "rdkit2d": Rdkit2d()
}

def one_prediction(mol):
    preds = {}
    for dn in necessary_descriptors:
        fp = descriptor_calculators[dn]
        X = fp.calc([mol])
        for tn in necessary_tasks:
            sess = sessions[(dn, tn)]
            if tn == "classification":
                pred = sess.predict_proba(X)[:,1]
            else:
                pred = sess.predict(X)[:]
            preds[(dn, tn)] = pred[0]
    return preds


if mol is None:
    y=0
    stdev=0
else:
    preds=one_prediction(mol)
    values=[]
    for k,v in preds.items():
        values += [v]
    y=np.mean(values)
    stdev=np.std(values)
col1.write("{0:.2f} +/- {1:.2f}".format(y,stdev))

col2.subheader("MW")
if mol is None:
    mw= 0
else:
    mw=ExactMolWt(mol)
col2.write("{0:.2f}".format(mw))

col3.subheader("SLogP")
if mol is None:
    logp = 0
else:
    logp=MolLogP(mol)
col3.write("{0:.2f}".format(logp))

col4.subheader("QED")
if mol is None:
    qed_score = 0.00
else:
    qed_score=qed(mol)
col4.write("{0:.2f}".format(qed_score))

col5.subheader("S4 Sim")
df = pd.read_csv("data/series4_processed.csv") #get series4 molecules for tanimoto similarity
s4_smiles = df["smiles"].tolist()
s4_mols=[Chem.MolFromSmiles(smi) for smi in s4_smiles]
ref_fps=mols_to_fingerprints(s4_mols)
if mol is None:
    tan=0
else:
    query_fp = mols_to_fingerprints([mol])
    tan=np.array([np.max(BulkTanimotoSimilarity(fp, ref_fps)) for fp in query_fp])[0]
col5.write("{0:.2f}".format(tan))

col6.subheader("RAscore")
if mol is None:
    ra=0
else:
    ra_fp=ra_fingerprint(mol)
    ra_fp = np.array([ra_fp], dtype=np.float32)
    sess = rt.InferenceSession("model/ra_model.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[1].name
    ra = sess.run([label_name], {input_name: ra_fp})[0][0][1]
col6.write("{0:.2f}".format(ra))

# Third section: individual activity predictions

if mol is not None:
    st.header("Activity Predictions")
    col1, col2, col3, col4, col5, col6 =st.beta_columns(6)
    the_cols = [col1, col2, col3, col4, col5, col6]

    i = 0
    for task in ["classification", "regression"]:
       for n, desc in enumerate(["ecfp", "rdkit2d", "chembl"]):
           my_col = the_cols[i]
           pred = preds[(desc, task)]
           my_col.subheader("{0}{1}".format(task[0].upper(), n+1))
           my_col.write("{0:.2f}".format(pred))
           i += 1

#Fourth section: details on each prediction
st.header("Property details")

#activity
st.subheader("Activity prediction")
st.write("Average score across all predictors (3 classifiers and 3 regressors) +/- standard deviation. Individual scores are detailed in the Activity Predictions row")
st.text("Interpretation: 0 = no activity - 1 = maximum activity")

#Molecular weight
st.subheader("Molecular Weight")
st.write("Calculation of the exact molecular weight using rdkit")
st.text("Interpretation: for series 4 molecules, MW should be around 450 Da")

#Solubility
st.subheader("Solubility")
st.write("Octanol/water partition coefficient estimation (LogP) as a measure of the compound solubility using rdkit ")
st.text("Interpretation: smaller values = more solubility. For drug-like molecules should be LogP<4")

#QED
st.subheader("Quantitative Estimation of Drug-like (QED)")
st.write("Estimation of the drug likeness using the QED score as calculated by rdkit")
st.text("Interpretation: 0 = lowest probability of being a drug - 1 = highest probability of being a drug")

#Series 4 similarity
st.subheader("Series4 similarity")
st.write("Maximum tanimoto similarity of the input molecule to the original series 4 molecules")
st.text("Interpretation: 0 = completely different - 1 = very similar to existing molecules")

#RA_score
st.subheader("Retrosynthetic Accessibility (RA) score")
st.write("Estimation of the synthetic feasibility for the input molecule as determined from the predictions of the computer aided synthesis planning tool AiZynthFinder. Developed by the [Reymond Group](https://github.com/reymond-group/RAscore)")
st.text("Interpretation: 0 = no synthetic route identified - 1 = synthetic route available")
