import pandas as pd
import logging

def process_and_aggregate(df):
    try:
        logging.info(f"Processing and aggregating {len(df)} records.")
        df['Minutes1DK'] = pd.to_datetime(df['Minutes1DK'])
        df.dropna(inplace=True)
        df['Minutes1DK'] = df['Minutes1DK'].dt.floor('h')
        df = df.groupby('Minutes1DK').mean().reset_index()

        full_range = pd.date_range(start=df['Minutes1DK'].min(), end=df['Minutes1DK'].max(), freq='h')
        df.set_index('Minutes1DK', inplace=True)
        df = df.reindex(full_range)
        df.interpolate(method='linear', inplace=True)
        df.dropna(inplace=True)
        df = df.round(2)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'TimeStamp'}, inplace=True)

        logging.info(f"Aggregated data to {len(df)} records.")
        return df

    except Exception as e:
        logging.error(f"Error during data aggregation: {e}")
        return pd.DataFrame()
