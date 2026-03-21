from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from API.key.auth import verify_api_key, get_password_hash, API_KEYS_DB
from API.key.keygen import generate_api_key
import secrets
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import os

app = FastAPI(title="Building Energy Anomaly Detection API", version="1.0.0")

@app.get("/open")
def open_endpoint():
    return {"message": "This is an open endpoint."}

@app.get("/protected", dependencies=[Depends(verify_api_key)])
def protected_endpoint():
    return {"message": "You used a valid API key!"}

@app.post("/admin/create-key", dependencies=[Depends(verify_api_key)])
def create_new_api_key():
    new_key = generate_api_key()
    
    return {"message": "New API key generated and stored.", "api_key": new_key}

class AnomalySummary(BaseModel):
    total_points: int
    anomaly_count: int
    anomaly_percentage: float
    votes_distribution: dict
    top_anomalies: list

def preprocess_data(df_path: str):
    """Load and preprocess data from CSV"""
    df = pd.read_csv(df_path)
    
    if df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if not df['timestamp'].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'timestamp' in numeric_cols:
        numeric_cols.remove('timestamp')
    
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found!")
    
    X = df[numeric_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return df, X, numeric_cols

def train_isolation_forest(X):
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    return iso_forest.fit_predict(X), iso_forest

def train_local_outlier_factor(X):
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.05,
        n_jobs=-1
    )
    return lof.fit_predict(X)

def train_elliptic_envelope(X):
    n_samples = len(X)
    if n_samples < 10:
        return np.ones(n_samples, dtype=int), None
    
    n_features = X.shape[1]
    if n_samples < 100 or n_features > 500:
        support_fraction = 0.5
    elif n_samples < 500 or n_features > 1000:
        support_fraction = 0.6
    else:
        support_fraction = 0.7
    
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        robust_cov = EllipticEnvelope(
            contamination=0.05,
            random_state=42,
            support_fraction=support_fraction
        )
        anomaly_pred = robust_cov.fit_predict(X_scaled)
        return anomaly_pred, robust_cov
    except Exception as e:
        print(f"EllipticEnvelope failed: {e}")
        return None, None

@app.post("/detect-anomalies", response_model=AnomalySummary)
async def detect_anomalies():
    data_path = "data/meters/whole/eda.csv"
    if not os.path.exists(data_path):
        return JSONResponse(status_code=404, content={"error": f"Data file not found: {data_path}"})
    
    try:
        df, X, numeric_cols = preprocess_data(data_path)
        
        # Train models
        results = Parallel(n_jobs=-1)(delayed(train_isolation_forest)(X) for _ in [1])
        anomaly_iso, iso_forest = results[0]
        
        anomaly_lof = train_local_outlier_factor(X)
        
        anomaly_maha_result = train_elliptic_envelope(X)
        if anomaly_maha_result[0] is None:
            anomaly_maha = anomaly_iso
        else:
            anomaly_maha, _ = anomaly_maha_result
        
        # Compute votes
        df['anomaly_votes'] = (
            (anomaly_iso == -1).astype(int) + 
            (anomaly_lof == -1).astype(int) + 
            (anomaly_maha == -1).astype(int)
        )
        df['is_anomaly'] = (df['anomaly_votes'] >= 2).astype(int)
        
        # Summary
        total_points = len(df)
        anomaly_df = df[df['is_anomaly'] == 1]
        anomaly_count = len(anomaly_df)
        percentage = (anomaly_count / total_points * 100) if total_points > 0 else 0
        
        votes_dist = df['anomaly_votes'].value_counts().sort_index().to_dict()
        
        # Top anomalies (first 10)
        top_anomalies = anomaly_df.head(10).to_dict('records')
        
        return AnomalySummary(
            total_points=total_points,
            anomaly_count=anomaly_count,
            anomaly_percentage=round(percentage, 2),
            votes_distribution=votes_dist,
            top_anomalies=top_anomalies
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "Energy Anomaly Detection API", "endpoints": ["/detect-anomalies (POST)"]}


