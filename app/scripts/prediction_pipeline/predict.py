import sys
from datetime import datetime
import logging
import os
from scripts.prediction_pipeline.updater import update_actuals
from scripts.prediction_pipeline.predictor import run_prediction_pipeline, get_latest_model_info, evaluate_latest_predictions
from scripts.prediction_pipeline.retrainer import retrain_model

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

RETRAIN_THRESHOLD = 65.0

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=os.path.join("logs", "prediction_pipeline.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def predict():
    update_actuals()
    evaluate_latest_predictions() 

    _, _, pseudo_acc = get_latest_model_info()
    if pseudo_acc is None or pseudo_acc < RETRAIN_THRESHOLD:
        logging.info("🚨 Pseudo accuracy too low. Retraining triggered.")
        retrain_model()

    run_prediction_pipeline()

if __name__ == "__main__":

    print(f"🟡 Starting prediction pipeline at {datetime.now()}", file=sys.stdout)
    predict()
    print(f"✅ Finished prediction pipeline at {datetime.now()}", file=sys.stdout)