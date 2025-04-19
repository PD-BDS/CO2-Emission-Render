import sqlite3
import pandas as pd
from scripts.data_pipeline.config import DB_PATH
import logging

def store_to_db(aggregated_df, features_df):
    try:
        logging.info("Storing data to database.")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Convert datetime columns to string format
        aggregated_df['TimeStamp'] = aggregated_df['TimeStamp'].astype(str)

        # Ensure no NaN values are in the data
        aggregated_df.fillna(0, inplace=True)  # Replace NaN with 0 for simplicity, or choose appropriate strategy

        # Store aggregated data
        aggregated_records = len(aggregated_df)
        aggregated_df.to_sql('aggregated_data', conn, if_exists='append', index=False)

        # Ensure engineered features have no NaN values and convert TimeStamp to string
        features_df['TimeStamp'] = features_df['TimeStamp'].astype(str)
        features_df.fillna(0, inplace=True)

        # Store engineered features
        features_records = len(features_df)
        features_df = features_df[[
            'TimeStamp',
            'CO2_lag_1',
            'CO2_lag_2',
            'CO2_lag_3',
            'CO2_lag_4',
            'CO2_lag_5',
            'CO2_rolling_mean_rolling_window_6',
            'CO2_rolling_std_rolling_window_6',
            'CO2_rolling_mean_rolling_window_12',
            'CO2_rolling_std_rolling_window_12'
        ]]

        features_df.to_sql('engineered_features', conn, if_exists='append', index=False)

        # Log the new data insertion
        log_description = f"Inserted {aggregated_records} aggregated records and {features_records} engineered feature records."
        cursor.execute('''INSERT INTO new_data_log (record_count, log_description) VALUES (?, ?)''', 
                       (aggregated_records + features_records, log_description))

        conn.commit()
        conn.close()

        logging.info("Data successfully stored in the database and log updated.")

    except sqlite3.DatabaseError as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during database storage: {e}")

def get_last_date_from_db():
    try:
        logging.info("Fetching the last date from the database.")
        conn = sqlite3.connect(DB_PATH)
        last_date_query = "SELECT MAX(TimeStamp) AS last_date FROM aggregated_data"
        last_date_df = pd.read_sql(last_date_query, conn)
        last_date = last_date_df['last_date'].iloc[0] if not last_date_df.empty else None
        conn.close()

        if last_date:
            logging.info(f"Last date from the database: {last_date}")
        else:
            logging.info("No date found in the database.")
        
        return last_date

    except sqlite3.DatabaseError as e:
        logging.error(f"Database error: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while fetching last date: {e}")
        return None
