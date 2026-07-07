import pandas as pd

def load_movie_data(filepath='data/movies.csv'):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        return None
