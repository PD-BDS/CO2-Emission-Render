from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
import os

# Define path to the SQLite database
DB_PATH = os.path.join(os.getcwd(), "database", "co2_emission.db")

# Create FastAPI app
app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router with prefix
api_router = APIRouter()

# Utility function to query the database
def query_db(query: str, params=()):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    # Replace problematic float values with None (which becomes null in JSON)
    df = df.replace([float('inf'), float('-inf')], None)
    df = df.where(pd.notnull(df), None)  # Replace NaNs with None

    return df.to_dict(orient="records")


# Endpoint: Last 24h CO₂ emissions
@api_router.get("/last-24h-emissions")
def get_last_24h_emissions():
    return query_db("""
        SELECT TimeStamp, CO2Emission FROM aggregated_data
        WHERE TimeStamp >= datetime('now', '-1 day')
        ORDER BY TimeStamp
    """)

# Endpoint: Next 6h predictions
@api_router.get("/next-6h-predictions")
def get_next_6h_predictions():
    return query_db("""
        SELECT TimeStamp, Prediction FROM predictions
        WHERE TimeStamp > datetime('now')
        ORDER BY TimeStamp
        LIMIT 6
    """)

# Endpoint: Last 6h predictions vs actual
@api_router.get("/last-6h-predictions-vs-actual")
def get_last_6h_predictions_vs_actual():
    return query_db("""
        SELECT TimeStamp, Prediction, Actual FROM predictions
        WHERE TimeStamp <= datetime('now')
        ORDER BY TimeStamp DESC
        LIMIT 6
    """)

# Endpoint: Best model details
@api_router.get("/best-model")
def get_latest_model():
    return query_db("""
        SELECT m.Model_id, m.Model_name, m.Version, m.Created_at,
               e.Pseudo_accuracy, e.RMSE, e.MAE, e.R2, e.MAPE
        FROM model_table m
        JOIN model_evaluations e ON m.Model_id = e.model_id
        ORDER BY m.Created_at DESC
        LIMIT 1
    """)

# Register the API router under the /api prefix
app.include_router(api_router, prefix="/api")
