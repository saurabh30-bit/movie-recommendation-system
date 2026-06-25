import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.similarity_matrix = None
        self.indices = None

    def load_data(self):
        # Mock dataset for demonstration
        # In a real scenario, this would load a CSV like the TMDB 5000 Movies Dataset
        data = {
            'id': [1, 2, 3, 4, 5],
            'title': [
                'The Matrix', 
                'Inception', 
                'Interstellar', 
                'The Notebook', 
                'Titanic'
            ],
            'tags': [
                'action sci-fi hackers simulation neo',
                'action sci-fi dreams thieves heist',
                'sci-fi space time-travel astronauts physics',
                'romance drama love story memory',
                'romance drama ship iceberg love'
            ]
        }
        self.movies_df = pd.DataFrame(data)
        
        # Create a Series for reverse mapping of movie titles to indices
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title']).drop_duplicates()

    def train_model(self):
        print("Training Content-Based Filtering Model...")
        # Create TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Fit and transform the 'tags' column to create the feature matrix
        tfidf_matrix = tfidf.fit_transform(self.movies_df['tags'])
        
        # Compute the cosine similarity matrix
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print("Model Training Complete.")

    def get_recommendations(self, title, top_n=2):
        if title not in self.movies_df['title'].values:
            return f"Movie '{title}' not found in the database."

        # Get the index of the movie that matches the title
        idx = self.indices[title]

        # Get pairwise similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort the movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the top N most similar movies (ignoring the first one, which is itself)
        sim_scores = sim_scores[1:top_n+1]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top N most similar movies
        return self.movies_df['title'].iloc[movie_indices].tolist()


if __name__ == "__main__":
    recommender = MovieRecommender()
    recommender.load_data()
    recommender.train_model()

    print("\n--- Testing Recommendations ---")
    
    test_movie = 'The Matrix'
    print(f"\nIf you liked '{test_movie}', you might also like:")
    recommendations = recommender.get_recommendations(test_movie)
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie}")

    test_movie = 'The Notebook'
    print(f"\nIf you liked '{test_movie}', you might also like:")
    recommendations = recommender.get_recommendations(test_movie)
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie}")
