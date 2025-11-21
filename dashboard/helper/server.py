from fastapi import FastAPI, Request
import uvicorn
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI()

# ---------- Load ML Model & Scaler ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler_new_xgb.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "models", "xgboost_model_new.pkl"))

label_map = {0: "Web", 1: "Multimedia", 2: "Social Media", 3: "Malicious"}

column_mapping = {
    'FlowDuration': 'duration',
    'TotalFwdIAT': 'total_fiat',
    'TotalBwdIAT': 'total_biat',
    'FwdIATMin': 'min_fiat',
    'BwdIATMin': 'min_biat',
    'FwdIATMax': 'max_fiat',
    'BwdIATMax': 'max_biat',
    'FwdIATMean': 'mean_fiat',
    'BwdIATMean': 'mean_biat',
    'PktsPerSec': 'flowPktsPerSecond',
    'BytesPerSec': 'flowBytesPerSecond',
    'FlowIATMin': 'min_flowiat',
    'FlowIATMax': 'max_flowiat',
    'FlowIATMean': 'mean_flowiat',
    'FlowIATStd': 'std_flowiat',
    'MinActive': 'min_active',
    'MeanActive': 'mean_active',
    'MaxActive': 'max_active',
    'StdActive': 'std_active',
    'MinIdle': 'min_idle',
    'MeanIdle': 'mean_idle',
    'MaxIdle': 'max_idle',
    'StdIdle': 'std_idle'
}

model_features = list(column_mapping.values())

# ---------- Classification Function ----------
def classify_flow(flow):
    df = pd.DataFrame([flow])

    df = df.rename(columns=column_mapping)
    df = df.reindex(columns=model_features, fill_value=0)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    X = scaler.transform(df)
    pred = model.predict(X)[0]
    return label_map[pred]

# ---------- API ----------
@app.post("/api/flows")
async def receive_flow(request: Request):
    flow = await request.json()

    if "row_id" not in flow:
        return {"error": "row_id is required"}

    predicted_class = classify_flow(flow)

    return {
        "row_id": flow["row_id"],
        "predicted_class": predicted_class,
        "status": "classified"
    }

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
