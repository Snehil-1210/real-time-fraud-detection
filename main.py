
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np, time, redis, json

app    = FastAPI(title='Fraud Detection API')
model  = joblib.load('model/fraud_model.pkl')
scaler = joblib.load('model/scaler.pkl')
try: r = redis.Redis(host='localhost', port=6379, decode_responses=True)
except: r = None

class Transaction(BaseModel):
    time: float
    v1: float; v2: float; v3: float; v4: float; v5: float
    v6: float; v7: float; v8: float; v9: float; v10: float
    v11: float; v12: float; v13: float; v14: float; v15: float
    v16: float; v17: float; v18: float; v19: float; v20: float
    v21: float; v22: float; v23: float; v24: float; v25: float
    v26: float; v27: float; v28: float
    amount: float

@app.post('/predict')
def predict(tx: Transaction):
    features = np.array([[tx.time, tx.v1, tx.v2, tx.v3, tx.v4, tx.v5,
        tx.v6, tx.v7, tx.v8, tx.v9, tx.v10, tx.v11, tx.v12, tx.v13,
        tx.v14, tx.v15, tx.v16, tx.v17, tx.v18, tx.v19, tx.v20,
        tx.v21, tx.v22, tx.v23, tx.v24, tx.v25, tx.v26, tx.v27,
        tx.v28, tx.amount]])
    features[:, [0, 29]] = scaler.transform(features[:, [0, 29]])
    prob = model.predict_proba(features)[0][1]
    is_fraud = bool(prob > 0.5)
    result = {'fraud': is_fraud, 'probability': round(float(prob), 4),
              'risk': 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.4 else 'LOW',
              'timestamp': time.time()}
    if r: r.lpush('fraud_results', json.dumps(result)); r.ltrim('fraud_results', 0, 999)
    return result
