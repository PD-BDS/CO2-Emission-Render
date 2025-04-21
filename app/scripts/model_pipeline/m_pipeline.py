import os
import torch
import joblib
import logging
import hashlib
import sqlite3
import numpy as np
from datetime import datetime

from scripts.model_pipeline.data_utils import load_data, create_sequences
from scripts.model_pipeline.trainer import train_model
from scripts.model_pipeline.evaluator import evaluate_model
from scripts.model_pipeline.logger import log_to_database
from scripts.model_pipeline.model_definitions import AttentionLSTMModel


os.makedirs("logs", exist_ok=True) 

# Setup logging
logging.basicConfig(
    filename='logs/training_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
DB_PATH = "database/co2_emission.db"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs("models", exist_ok=True)

TRAIN_START = "2023-01-01 00:00:00"
TRAIN_END = "2025-04-01 23:00:00"

BEST_PARAMS = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'lr': 0.001
}

def get_model_identity(best_params, db_path):
    param_string = f"{best_params['hidden_size']}_{best_params['num_layers']}_{best_params['dropout']}_{best_params['lr']}"
    model_base = f"LSTM_Attn_H{best_params['hidden_size']}_L{best_params['num_layers']}_LR{best_params['lr']}"
    version = "v1.0"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT Version FROM model_table WHERE Model_name = ?", (model_base,))
    versions = cursor.fetchall()
    conn.close()

    if versions:
        version_nums = [float(v[0].replace('v','')) for v in versions if v[0].startswith('v')]
        new_version_num = max(version_nums) + 0.1
        version = f"v{new_version_num:.1f}"

    return model_base, version

MODEL_NAME, VERSION = get_model_identity(BEST_PARAMS, DB_PATH)
MODEL_PATH = f"models/{MODEL_NAME}_{VERSION}_{timestamp}.pth"
dataset_label = f"test_{timestamp}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    logging.info("🚀 Pipeline started...")

    input_window = 24
    output_window = 6
    train_range = f"{TRAIN_START} to {TRAIN_END}"

    # Load and preprocess data
    timestamps, X_scaled, y_scaled, scaler_x, scaler_y = load_data()
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, input_window, output_window)

    # Train/val/test split
    X_train, X_val, X_test = (
        X_seq[:int(0.7*len(X_seq))],
        X_seq[int(0.7*len(X_seq)):int(0.85*len(X_seq))],
        X_seq[int(0.85*len(X_seq)):] 
    )
    y_train, y_val, y_test = (
        y_seq[:int(0.7*len(y_seq))],
        y_seq[int(0.7*len(y_seq)):int(0.85*len(y_seq))],
        y_seq[int(0.85*len(y_seq)):] 
    )

    test_timestamps = timestamps[input_window + output_window + int(0.85 * len(y_seq)) : input_window + output_window + len(y_seq)]

    # Train model
    model = train_model(
        X_train, y_train, X_val, y_val,
        input_size=X_seq.shape[2],
        output_window=output_window,
        best_params=BEST_PARAMS,
        model_path=MODEL_PATH
    )

    model.load_state_dict(torch.load(MODEL_PATH))
    joblib.dump(scaler_x, "models/scaler_x.pkl")
    joblib.dump(scaler_y, "models/scaler_y.pkl")
    logging.info("Model and Scalers saved.")

    # Evaluate model
    y_pred, y_true, mae, mse, rmse, r2, mape, acc, ts = evaluate_model(
        model, X_test, y_test, scaler_y, test_timestamps, output_window
    )

    metrics = {
        "mae": mae, "mse": mse, "rmse": rmse,
        "r2": r2, "mape": mape, "accuracy": acc
    }

    # Log to DB
    log_to_database(
        DB_PATH, MODEL_PATH,
        model_name=MODEL_NAME,
        version=VERSION,
        train_range=train_range,
        best_params=BEST_PARAMS,
        y_pred=y_pred,
        y_true=y_true,
        metrics=metrics,
        timestamps=ts,
        dataset_label=dataset_label
    )

    logging.info("🏁 Pipeline finished successfully.")
