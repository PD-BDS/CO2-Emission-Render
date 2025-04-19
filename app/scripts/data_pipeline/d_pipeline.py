import sys
from datetime import datetime, timedelta
import logging
from scripts.data_pipeline.db_operation import get_last_date_from_db, store_to_db
from scripts.data_pipeline.data_fetcher import fetch_data
from scripts.data_pipeline.data_processor import process_and_aggregate
from scripts.data_pipeline.feature_engineering import add_features

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def run_pipeline():
    try:
        # Check the last date stored in the database
        last_date = get_last_date_from_db()

        # If last date exists, extract the next 24 hours of data
        if last_date:
            start_date = (datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S') + timedelta(hours=1))
            end_date = datetime.now()
        else:
            # Otherwise, extract the full range of initial data
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2025, 4, 9)

        logging.info(f"Fetching data from {start_date} to {end_date}")

        # Fetch and process the data
        raw_df = fetch_data(start_date, end_date)
        if raw_df.empty:
            logging.warning("No data fetched. Skipping pipeline execution.")
            return

        aggregated_df = process_and_aggregate(raw_df)
        features_df = add_features(aggregated_df)

        if not aggregated_df.empty and not features_df.empty:
            # Store the results in the database
            store_to_db(aggregated_df, features_df)

            # Optionally save to CSV for backup/inspection
            aggregated_df.to_csv("database/aggregated_hourly_data.csv", index=False)
            logging.info("Pipeline executed successfully.")
        else:
            logging.warning("No valid data after processing. Skipping storage.")

    except Exception as e:
        logging.error(f"Error during pipeline execution: {e}")

if __name__ == "__main__":
    print(f"🟢 Starting data pipeline at {datetime.now()}", file=sys.stdout)
    run_pipeline()
    print(f"✅ Finished data pipeline at {datetime.now()}", file=sys.stdout)
