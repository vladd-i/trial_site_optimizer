# src/data/data_loader.py

import pandas as pd
from configs.config import DATA_PATH 
from .preprocess import clean_target_data, clean_trial_site_data, clean_trial_data

def load_country_data():
    """Load country data from a Parquet file."""
    df = pd.read_parquet(f"{DATA_PATH}/country.parquet")
    return df

def load_target_data():
    """Load target data from a Parquet file."""
    df = pd.read_parquet(f"{DATA_PATH}/target.parquet")
    df_cleaned = clean_target_data(df)
    return df_cleaned

def load_trial_site_data():
    """Load trial site data from a Parquet file."""
    df = pd.read_parquet(f"{DATA_PATH}/trial_site.parquet")
    df_cleaned = clean_trial_site_data(df)
    return df_cleaned

def load_trial_data():
    """Load trial data from a Parquet file."""
    df = pd.read_parquet(f"{DATA_PATH}/trial.parquet")
    df_cleaned = clean_trial_data(df)
    return df_cleaned
