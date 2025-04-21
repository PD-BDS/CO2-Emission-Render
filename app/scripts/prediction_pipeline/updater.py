import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "database" / "co2_emission.db")

def update_actuals():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE predictions
        SET Actual = (
            SELECT CO2Emission FROM aggregated_data ad
            WHERE ad.TimeStamp = predictions.TimeStamp
        )
        WHERE Actual IS NULL
    ''')
    conn.commit()
    conn.close()
