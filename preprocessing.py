import pandas as pd

def normalize_ratings(df, col='rating'):
    min_val = df[col].min()
    max_val = df[col].max()
    df[col + '_norm'] = (df[col] - min_val) / (max_val - min_val)
    return df

def remove_sparse_users(df, min_ratings=5):
    user_counts = df['userId'].value_counts()
    active_users = user_counts[user_counts >= min_ratings].index
    return df[df['userId'].isin(active_users)]
