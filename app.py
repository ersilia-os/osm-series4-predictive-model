import streamlit as st
import joblib
import pandas as pd
import numpy as np
import onnxruntime as rt
import json
import os, sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Descriptors import MolLogP
from rdkit.Chem import Draw
from rdkit.DataStructs import BulkTanimotoSimilarity

sys.path.append(os.path.dirname(__file__))
import utils.SA_Score.sascorer as sascorer
from src.fingerprinter import get_fingerprints, mols_to_fingerprints, ra_fingerprint
from src.eosdescriptors.chembl import Chembl
from src.eosdescriptors.ecfp import Ecfp
from src.eosdescriptors.rdkit2d import Rdkit2d
from src.eosdescriptors.rdkitfpbits import RdkitFpBits


# APP:
st.set_page_config(page_title=None, page_icon=None, layout='wide', initial_sidebar_state='auto')

st.title("OSM series 4 calculator")
st.write("See [GitHub Repo](https://github.com/ersilia-os/osm-series4-predictive-model) | Check GitHub OSM Issue [#34](https://github.com/OpenSourceMalaria/Series4_PredictiveModel/issues/34) | Learn about [Ersilia Open Source Initiative](https://ersilia.io)")

# First section: molecule input and drawing
col1, col2, col3 = st.beta_columns(3)
# Input
col1.subheader("Input SMILES")
smiles = col1.text_input("Input molecule in SMILES format.", value = "BrC(F)Oc1ccc(-c2nnc3cncc(CN4Cc5ccccc5C4)n23)cc1")
if not smiles:
    mol = None
else:
    mol=Chem.MolFromSmiles(smiles)

# Image
col2.subheader("Input molecule")
if mol is not None:
    col2.image(Draw.MolToImage(mol), width=200)

# Closest molecule
col3.subheader("Closest known S4")

@st.cache
def get_s4_mols_to_fingerprints():
    df = pd.read_csv("data/series4_processed.csv") #get series4 molecules for tanimoto similarity
    s4_smiles = df["smiles"].tolist()
    s4_mols = [Chem.MolFromSmiles(smi) for smi in s4_smiles]
    s4_osm = df["osm"].tolist()
    ref_fps=mols_to_fingerprints(s4_mols)
    return s4_smiles, s4_mols, s4_osm, ref_fps

s4_smiles, s4_mols, s4_osm, ref_fps = get_s4_mols_to_fingerprints()

if mol is None:
    pass
else:
    query_fp = mols_to_fingerprints([mol])
    fp = query_fp[0]
    bulk_sims = BulkTanimotoSimilarity(fp, ref_fps)
    idx = np.argmax(bulk_sims)
    s4_mol = s4_mols[idx]
    col3.image(Draw.MolToImage(s4_mol), width=200)
    col3.text(s4_osm[idx])

# Second section: prediction of properties
st.header("Properties")
col1, col2, col3, col4, col5, col6, col7, col8 = st.beta_columns(8)

col1.subheader("Activity")
MODELS_DIR = "model"

@st.cache
def get_models_files():
    models_files = {}
    with open(os.path.join(MODELS_DIR, "models.json"), "r") as f:
        all_models = json.load(f)
    for dn, tn in all_models:
        models_files[(dn, tn)] = os.path.join(MODELS_DIR, "{0}_{1}.pkl".format(dn, tn))
    return all_models, models_files

all_models, models_files = get_models_files()

@st.cache(allow_output_mutation=True)
def get_sessions():
    sessions = {}
    for k, v in models_files.items():
        sess = joblib.load(v)
        sessions[k] = sess
    return sessions

sessions = get_sessions()

necessary_descriptors = sorted(set([k[0] for k in all_models]))
necessary_tasks = sorted(set([k[1] for k in all_models]))

@st.cache
def get_descriptor_calculators():
    descriptor_calculators = {
        "chembl": Chembl(),
        "ecfp": Ecfp(),
        "rdkit2d": Rdkit2d(),
        "rdkitfpbits": RdkitFpBits()
    }
    return descriptor_calculators

descriptor_calculators = get_descriptor_calculators()

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
    tan=np.max(bulk_sims)
col5.write("{0:.2f}".format(tan))

col6.subheader("RA score")

@st.cache
def get_rascore_inference_session():
    sess = rt.InferenceSession("model/ra_model.onnx")
    return sess

if mol is None:
    ra=0
else:
    ra_fp=ra_fingerprint(mol)
    ra_fp = np.array([ra_fp], dtype=np.float32)
    sess = get_rascore_inference_session()
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[1].name
    ra = sess.run([label_name], {input_name: ra_fp})[0][0][1]
col6.write("{0:.2f}".format(ra))

col7.subheader("SA score")
if mol is None:
    sa=0
else:
    sa = sascorer.calculateScore(mol)
col7.write("{0:.2f}".format(sa))

col8.subheader("HDB & HDA")
if mol is None:
    comp = 0
else:
    HDonorSmarts = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')
    HAcceptorSmarts = Chem.MolFromSmarts('[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
                                         '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
                                         '$([nH0,o,s;+0])]')
    def count_hbd_hba_atoms(m):
        HDonor = m.GetSubstructMatches(HDonorSmarts)
        HAcceptor = m.GetSubstructMatches(HAcceptorSmarts)
        return len(set(HDonor + HAcceptor))
    comp = count_hbd_hba_atoms(mol)
    col8.write("{0}".format(comp))

# Third section: individual activity predictions

if mol is not None:
    st.header("Activity Predictions")
    the_cols = st.beta_columns(8)
    i = 0
    for task in ["classification", "regression"]:
       for n, desc in enumerate(["ecfp", "rdkitfpbits", "rdkit2d", "chembl"]):
           my_col = the_cols[i]
           pred = preds[(desc, task)]
           my_col.subheader("{0}{1}".format(task[0].upper(), n+1))
           my_col.write("{0:.2f}".format(pred))
           i += 1

#Fourth section: details on each prediction
st.header("Property details")

#activity
st.subheader("Activity prediction")
st.write("Average score across all predictors (4 calibrated classifiers (C) and 4 regressors (R) obtained with different descriptors) +/- standard deviation across estimators. Individual scores are detailed in the Activity Predictions row.")
st.text("Interpretation: 0 = no activity - 1 = maximum activity")

#Molecular weight
st.subheader("Molecular Weight")
st.write("Calculation of the exact molecular weight using Rdkit.")
st.text("Interpretation: for series 4 molecules, MW should be around 450 Da")

#Solubility
st.subheader("SLogP")
st.write("Octanol/water partition coefficient estimation (LogP) as a measure of the molecule solubility using Rdkit.")
st.text("Interpretation: smaller values = more solubility. For drug-like molecules should be LogP<4")

#QED
st.subheader("Quantitative Estimation of Drug-like (QED)")
st.write("Estimation of the drug likeness using the QED score as calculated by Rdkit.")
st.text("Interpretation: 0 = lowest probability of being a drug - 1 = highest probability of being a drug")

#Series 4 similarity
st.subheader("Series4 similarity")
st.write("Maximum tanimoto similarity of the input molecule to the original series 4 molecules.")
st.text("Interpretation: 0 = completely different - 1 = very similar to existing molecules")

#RA_score
st.subheader("Retrosynthetic Accessibility (RA) score")
st.write("Estimation of the synthetic feasibility for the input molecule as determined from the predictions of the computer aided synthesis planning tool AiZynthFinder. Developed by the [Reymond Group](https://github.com/reymond-group/RAscore).")
st.text("Interpretation: 0 = no synthetic route identified - 1 = synthetic route available")

#SA_score
st.subheader("Synthetic Accessiblity (SA) score")
st.write("Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions.")
st.text("Interpretation: 1 = easy to synthesize - 10 = difficult to synthesize")

#HBD and HBA
st.subheader("Hydrogen bond donors (HBD) and acceptors (HBA")
st.write("Total number of HBD and HBA")
st.text("Interpretation: HBD + HBA")
