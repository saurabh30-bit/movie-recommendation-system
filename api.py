import requests
import os

def fetch_movie_details(movie_id):
    api_key = os.getenv('TMDB_API_KEY')
    if not api_key:
        return None
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}')
    return response.json() if response.status_code == 200 else None
