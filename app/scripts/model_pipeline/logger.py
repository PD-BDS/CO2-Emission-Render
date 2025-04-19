# logger.py
import sqlite3
import hashlib
import logging

def log_to_database(DB_PATH, MODEL_PATH, model_name, version, train_range, best_params,
                    y_pred, y_true, metrics, timestamps, dataset_label):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        model_hash = hashlib.sha256(open(MODEL_PATH, 'rb').read()).hexdigest()

        cursor.execute('''
            INSERT INTO model_table (
                Model_name, Hidden_size, Num_layers, Dropout_rate, Learning_rate,
                Version, Trained_on, Model_path, Model_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            best_params['hidden_size'],
            best_params['num_layers'],
            best_params['dropout'],
            best_params['lr'],
            version,
            train_range,
            MODEL_PATH,
            model_hash
        ))
        model_id = cursor.lastrowid

        cursor.execute('''
            INSERT INTO model_training_sets (model_id, time_frame)
            VALUES (?, ?)
        ''', (model_id, train_range))

        for ts, pred, actual in zip(timestamps, y_pred[:, 0], y_true[:, 0]):
            cursor.execute('''
                INSERT INTO predictions (Model_id, TimeStamp, Prediction, Actual)
                VALUES (?, ?, ?, ?)
            ''', (model_id, ts.strftime('%Y-%m-%d %H:%M:%S'), float(pred), float(actual)))

        cursor.execute('''
            INSERT INTO model_evaluations (
                model_id, dataset_label, RMSE, MAE, MSE, R2, MAPE, Pseudo_accuracy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_id,
            dataset_label,
            float(metrics['rmse']),
            float(metrics['mae']),
            float(metrics['mse']),
            float(metrics['r2']),
            float(metrics['mape']),
            float(metrics['accuracy'])
        ))

        conn.commit()
        logging.info(f"✅ All records inserted for model_id={model_id}")

    except Exception as e:
        conn.rollback()
        logging.error(f"❌ DB logging failed: {e}")
    finally:
        conn.close()
