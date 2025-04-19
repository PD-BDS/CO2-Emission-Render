from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.getcwd(), "database", "co2_emission.db")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def query_db(query: str, params=()):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df.to_dict(orient="records")

@app.get("/last-24h-emissions")
def get_last_24h_emissions():
    return query_db("""
        SELECT TimeStamp, CO2Emission FROM aggregated_data
        WHERE TimeStamp >= datetime('now', '-1 day')
        ORDER BY TimeStamp
    """)

@app.get("/next-6h-predictions")
def get_next_6h_predictions():
    return query_db("""
        SELECT TimeStamp, Prediction FROM predictions
        WHERE TimeStamp > datetime('now')
        ORDER BY TimeStamp
        LIMIT 6
    """)

@app.get("/last-6h-predictions-vs-actual")
def get_last_6h_predictions_vs_actual():
    return query_db("""
        SELECT TimeStamp, Prediction, Actual FROM predictions
        WHERE TimeStamp <= datetime('now')
        ORDER BY TimeStamp DESC
        LIMIT 6
    """)

@app.get("/best-model")
def get_latest_model():
    return query_db("""
        SELECT m.Model_id, m.Model_name, m.Version, m.Created_at, e.Pseudo_accuracy, e.RMSE, e.MAE, e.R2, e.MAPE
        FROM model_table m
        JOIN model_evaluations e ON m.Model_id = e.model_id
        ORDER BY m.Created_at DESC
        LIMIT 1
    """)
