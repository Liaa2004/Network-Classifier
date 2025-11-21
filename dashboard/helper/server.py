from fastapi import FastAPI, Request
import uvicorn
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import pandas as pd
import numpy as np
import joblib
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# ---------- MongoDB Setup ----------
MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client.flowdb
collection = db.flows

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Load Model + Scaler ----------
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

# ---------- Batch Buffer ----------
BATCH_SIZE = 10
flow_buffer = []
buffer_lock = asyncio.Lock()

# ---------- Classification ----------
async def classify_and_update(flows):
    if not flows:
        return

    df = pd.DataFrame(flows)
    row_ids = df["row_id"].tolist()

    df = df.rename(columns=column_mapping)
    df = df.reindex(columns=model_features, fill_value=0)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    X = scaler.transform(df)
    preds = model.predict(X)
    pred_classes = [label_map[p] for p in preds]

    # Update MongoDB
    for row_id, cls in zip(row_ids, pred_classes):
        await collection.update_one(
            {"row_id": row_id},
            {"$set": {"predicted_class": cls}}
        )

    print(f"[ML] Classified {len(flows)} flows")

# ---------- Flush Loop ----------
async def periodic_flush():
    while True:
        await asyncio.sleep(5)
        async with buffer_lock:
            if flow_buffer:
                batch = flow_buffer.copy()
                flow_buffer.clear()
                await classify_and_update(batch)

# ---------- API ----------
@app.post("/api/flows")
async def receive_flow(request: Request):
    data = await request.json()

    if "row_id" not in data:
        return {"status": "error", "message": "row_id is required"}

    await collection.insert_one(data)
    print(f"[STORE] Flow stored: {data['row_id']}")

    async with buffer_lock:
        flow_buffer.append(data)
        if len(flow_buffer) >= BATCH_SIZE:
            batch = flow_buffer.copy()
            flow_buffer.clear()
            asyncio.create_task(classify_and_update(batch))

    return {"status": "ok", "received": data["row_id"]}

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(periodic_flush())

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
