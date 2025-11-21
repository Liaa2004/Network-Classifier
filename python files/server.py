
# from fastapi import FastAPI, Request
# import uvicorn
# from motor.motor_asyncio import AsyncIOMotorClient
# import asyncio
# import pandas as pd
# import numpy as np
# import joblib
# import os
# from load_dotenv import load_dotenv
# load_dotenv()
# import joblib
# import os
# app = FastAPI()

# # ---------- MongoDB Atlas Async Setup ----------
# MONGO_URI = os.getenv(
#     "MONGO_URI"
# )
# client = AsyncIOMotorClient(MONGO_URI)

# db = client.flowdb
# collection = db.flows
# script_dir = os.path.dirname(__file__)
# # ---------- Load XGBoost Model and Scaler ----------
# scaler_path = os.path.join(script_dir, "models", "scaler_new_xgb.pkl")
# model_path = os.path.join(script_dir, "models", "xgboost_model_new.pkl")

# scaler = joblib.load(scaler_path)
# model = joblib.load(model_path)
# label_map = {0: "Web", 1: "Multimedia", 2: "Social Media", 3: "Malicious"}

# column_mapping = {
#     'FlowDuration': 'duration',
#     'TotalFwdIAT': 'total_fiat',
#     'TotalBwdIAT': 'total_biat',
#     'FwdIATMin': 'min_fiat',
#     'BwdIATMin': 'min_biat',
#     'FwdIATMax': 'max_fiat',
#     'BwdIATMax': 'max_biat',
#     'FwdIATMean': 'mean_fiat',
#     'BwdIATMean': 'mean_biat',
#     'PktsPerSec': 'flowPktsPerSecond',
#     'BytesPerSec': 'flowBytesPerSecond',
#     'FlowIATMin': 'min_flowiat',
#     'FlowIATMax': 'max_flowiat',
#     'FlowIATMean': 'mean_flowiat',
#     'FlowIATStd': 'std_flowiat',
#     'MinActive': 'min_active',
#     'MeanActive': 'mean_active',
#     'MaxActive': 'max_active',
#     'StdActive': 'std_active',
#     'MinIdle': 'min_idle',
#     'MeanIdle': 'mean_idle',
#     'MaxIdle': 'max_idle',
#     'StdIdle': 'std_idle'
# }
# model_features = list(column_mapping.values())

# # ---------- Batch Buffer ----------
# BATCH_SIZE = 10
# flow_buffer = []
# buffer_lock = asyncio.Lock()  # ensures thread-safe access

# # ---------- Classification Function ----------
# async def classify_and_update(flows):
#     if not flows:
#         return
#     df = pd.DataFrame(flows)
#     row_ids = df["row_id"].tolist()

#     # Rename columns and select model features
#     df = df.rename(columns=column_mapping)
#     df = df.reindex(columns=model_features, fill_value=0)  # handle missing features
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df.fillna(0, inplace=True)

#     # Scale and predict
#     X_scaled = scaler.transform(df)
#     y_pred = model.predict(X_scaled)
#     predicted_classes = [label_map.get(p, p) for p in y_pred]

#     # Update MongoDB
#     for row_id, pred in zip(row_ids, predicted_classes):
#         await collection.update_one({"row_id": row_id}, {"$set": {"predicted_class": pred}})
    
#     print(f"[SERVER] Classified and updated {len(flows)} flows.")

# # ---------- Periodic Batch Flusher ----------
# async def periodic_flush():
#     while True:
#         await asyncio.sleep(5)  # every 5 seconds
#         async with buffer_lock:
#             if flow_buffer:
#                 batch = flow_buffer.copy()
#                 flow_buffer.clear()
#                 await classify_and_update(batch)

# # ---------- API Endpoint ----------
# @app.post("/api/flows")
# async def receive_flow(request: Request):
#     data = await request.json()
#     row_id = data.get("row_id")
#     if not row_id:
#         return {"status": "error", "message": "row_id is required"}

#     # Insert into MongoDB
#     await collection.insert_one(data)
#     print(f"[SERVER] Stored flow {row_id}")

#     # Add to buffer and trigger batch classification
#     async with buffer_lock:
#         flow_buffer.append(data)
#         if len(flow_buffer) >= BATCH_SIZE:
#             batch = flow_buffer.copy()
#             flow_buffer.clear()
#             asyncio.create_task(classify_and_update(batch))

#     return {"status": "ok", "received_row_id": row_id}

# # ---------- Startup Event ----------
# @app.on_event("startup")
# async def startup_event():
#     asyncio.create_task(periodic_flush())  # start periodic flush in background

# # ---------- Run ----------
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=5000)


