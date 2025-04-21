import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "database" / "co2_emission.db")

TRAIN_START = "2023-01-01 00:00:00"
TRAIN_END = "2025-04-01 23:00:00"

def load_data(start_date=None, end_date=None):
    conn = sqlite3.connect(DB_PATH)

    if start_date is None or end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_date = (datetime.now() - pd.DateOffset(years=2)).strftime("%Y-%m-%d %H:%M:%S")

    query = """
        SELECT * FROM (
            SELECT 
                a.TimeStamp,
                a.ProductionGe100MW,
                a.ProductionLt100MW,
                a.SolarPower,
                a.OffshoreWindPower,
                a.OnshoreWindPower,
                a.Exchange_Sum,
                a.CO2Emission,
                f.CO2_lag_1,
                f.CO2_lag_2,
                f.CO2_lag_3,
                f.CO2_lag_4,
                f.CO2_lag_5,
                f.CO2_rolling_mean_rolling_window_6,
                f.CO2_rolling_std_rolling_window_6,
                f.CO2_rolling_mean_rolling_window_12,
                f.CO2_rolling_std_rolling_window_12
            FROM aggregated_data a
            INNER JOIN engineered_features f ON a.TimeStamp = f.TimeStamp
            WHERE a.TimeStamp BETWEEN ? AND ?
            ORDER BY a.TimeStamp
        )
    """
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    conn.close()

    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df.set_index("TimeStamp", inplace=True)
    features = df.drop(columns=["CO2Emission"]).values
    target = df[["CO2Emission"]].values

    scaler_x = MinMaxScaler().fit(features)
    scaler_y = MinMaxScaler().fit(target)

    return df.index, scaler_x.transform(features), scaler_y.transform(target), scaler_x, scaler_y


def create_sequences(X, y, input_window=24, output_window=6):
    Xs, ys = [], []
    for i in range(len(X) - input_window - output_window):
        Xs.append(X[i:i+input_window])
        ys.append(y[i+input_window:i+input_window+output_window].flatten())
    return np.array(Xs), np.array(ys)