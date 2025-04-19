import requests
import pandas as pd
from scripts.data_pipeline.config import BASE_URL, RELEVANT_COLUMNS
import logging

def fetch_data(start, end):
    try:
        params = {
            "start": start.strftime('%Y-%m-%dT%H:%M'),
            "end": end.strftime('%Y-%m-%dT%H:%M')
        }
        logging.info(f"Fetching data from {start} to {end} using API.")
        response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
            logging.error(f"API request failed with status code {response.status_code}.")
            return pd.DataFrame()

        data = response.json().get("records", [])
        df = pd.DataFrame(data)

        if not df.empty:
            df = df[RELEVANT_COLUMNS]

        logging.info(f"Fetched {len(df)} records from the API.")
        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"Error during API request: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error in fetch_data: {e}")
        return pd.DataFrame()
