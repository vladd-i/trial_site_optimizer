# src/data/preprocess.py

def clean_target_data(df):
    """Cleans the target DataFrame by converting trial_id and site_id to integers."""
    df['trial_id'] = df['trial_id'].str.replace('trial_', '').astype(int)
    df['site_id'] = df['site_id'].str.replace('site_', '').astype(int)
    return df

def clean_trial_site_data(df):
    """Cleans the trial site DataFrame by converting trial_id and site_id to integers."""
    df['trial_id'] = df['trial_id'].str.replace('trial_', '').astype(int)
    df['site_id'] = df['site_id'].str.replace('site_', '').astype(int)
    return df

def clean_trial_data(df):
    """Cleans the trial DataFrame by converting trial_id and site_id to integers."""
    df['trial_id'] = df['trial_id'].str.replace('trial_', '').astype(int)
    return df
