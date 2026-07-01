from recommender import MovieRecommender

def test_recommender():
    rec = MovieRecommender()
    rec.load_data()
    rec.train_model()
    assert len(rec.movies_df) > 0
    print('Tests passed!')

if __name__ == '__main__':
    test_recommender()
