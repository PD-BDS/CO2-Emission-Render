import pandas as pd
import logging

def add_features(df):
    try:
        logging.info(f"Adding features to {len(df)} records.")
        df = df.copy()

        for i in range(1, 6):
            df[f'CO2_lag_{i}'] = df['CO2Emission'].shift(i)

        for window in [6, 12]:
            df[f'CO2_rolling_mean_rolling_window_{window}'] = df['CO2Emission'].rolling(window=window).mean()
            df[f'CO2_rolling_std_rolling_window_{window}'] = df['CO2Emission'].rolling(window=window).std()

        df.dropna(inplace=True)
        logging.info(f"Feature engineering completed with {len(df)} records.")
        return df

    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        return pd.DataFrame()
