
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Multi-Target QSAR Predictor",
              description="Predicts EGFR and BRAF inhibitory activity from SMILES, with applicability domain checking",
              version="2.1.0")

egfr_model = joblib.load("qsar_model.pkl")
braf_model = joblib.load("braf_model.pkl")
braf_train_fp = np.load("braf_train_fingerprints.npy")

AD_THRESHOLD = 0.5  # established from manual testing - see README

class MoleculeInput(BaseModel):
    smiles: str

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    desc = np.array([[
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
    ]])
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=1024)).reshape(1, -1)
    features = np.hstack([desc, fp])
    props = {
        "mw":       round(Descriptors.MolWt(mol), 2),
        "logp":     round(Descriptors.MolLogP(mol), 2),
        "hbd":      rdMolDescriptors.CalcNumHBD(mol),
        "hba":      rdMolDescriptors.CalcNumHBA(mol),
        "tpsa":     round(Descriptors.TPSA(mol), 2),
        "lipinski": bool(
            Descriptors.MolWt(mol) <= 500 and
            Descriptors.MolLogP(mol) <= 5 and
            rdMolDescriptors.CalcNumHBD(mol) <= 5 and
            rdMolDescriptors.CalcNumHBA(mol) <= 10
        )
    }
    return features, fp, props

def run_prediction(model, features, target_name):
    pred_label = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]
    confidence = round(float(pred_proba.max()), 3)
    prediction = "Active" if pred_label == 1 else "Inactive"
    return {"target": target_name, "prediction": prediction, "confidence": confidence}

def check_applicability_domain(fp, train_fp, threshold=AD_THRESHOLD):
    sim = cosine_similarity(fp, train_fp).max()
    return {
        "max_similarity": round(float(sim), 3),
        "in_domain": bool(sim >= threshold),
        "reliability": "Reliable" if sim >= threshold else "Low confidence - structurally dissimilar to training data"
    }

@app.get("/")
def root():
    return {"message": "Multi-target QSAR API running",
            "targets": ["EGFR", "BRAF"], "version": "2.1.0"}

@app.post("/predict")
def predict(molecule: MoleculeInput):
    features, fp, props = smiles_to_features(molecule.smiles)
    if features is None:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {molecule.smiles}")

    egfr_result = run_prediction(egfr_model, features, "EGFR")
    braf_result = run_prediction(braf_model, features, "BRAF")
    braf_ad = check_applicability_domain(fp, braf_train_fp)
    braf_result["applicability_domain"] = braf_ad

    return {
        "smiles": molecule.smiles,
        "properties": props,
        "predictions": {"EGFR": egfr_result, "BRAF": braf_result}
    }
