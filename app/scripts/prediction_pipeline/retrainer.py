import torch
import joblib
from datetime import datetime, timedelta
from scripts.model_pipeline.data_utils import load_data, create_sequences
from scripts.model_pipeline.trainer import train_model
from scripts.model_pipeline.evaluator import evaluate_model
from scripts.model_pipeline.logger import log_to_database
from scripts.model_pipeline.m_pipeline import BEST_PARAMS, get_model_identity

DB_PATH = "database/co2_emission.db"
MODEL_DIR = "models"
INPUT_WINDOW = 24
OUTPUT_WINDOW = 6

def retrain_model():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    end = datetime.now()
    start = end - timedelta(days=730)
    ts, X, y, scaler_x, scaler_y = load_data(start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S'))

    ts, X, y, scaler_x, scaler_y = load_data()
    X_seq, y_seq = create_sequences(X, y, INPUT_WINDOW, OUTPUT_WINDOW)

    train_size = int(0.85 * len(X_seq))
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]
    test_ts = ts[INPUT_WINDOW + OUTPUT_WINDOW + train_size : INPUT_WINDOW + OUTPUT_WINDOW + len(y_seq)]

    model_name, version = get_model_identity(BEST_PARAMS, DB_PATH)
    model_path = f"{MODEL_DIR}/{model_name}_{version}_{timestamp}.pth"

    model = train_model(X_train, y_train, X_test, y_test, X_seq.shape[2], OUTPUT_WINDOW, BEST_PARAMS, model_path)
    model.load_state_dict(torch.load(model_path))

    joblib.dump(scaler_x, f"{MODEL_DIR}/scaler_x.pkl")
    joblib.dump(scaler_y, f"{MODEL_DIR}/scaler_y.pkl")

    y_pred, y_true, mae, mse, rmse, r2, mape, acc, ts_eval = evaluate_model(model, X_test, y_test, scaler_y, test_ts, OUTPUT_WINDOW)
    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape, "accuracy": acc}

    train_range = f"{start.strftime('%Y-%m-%d %H:%M:%S')} to {end.strftime('%Y-%m-%d %H:%M:%S')}"

    log_to_database(DB_PATH, model_path, model_name, version, train_range,
                    BEST_PARAMS, y_pred, y_true, metrics, ts_eval, f"retrain_{timestamp}")
