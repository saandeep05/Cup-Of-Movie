import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecommenderSystem(object):
    def __init__(self, file_path, cols_list):
        self.movies_df = pd.read_csv(file_path)

        # defining_features are the columns on which the recommender system focuses on
        self.defining_features = cols_list

        # combined_features is the result of adding all the defining feature column for each movie
        self.combined_features = ''

        # filling null values in the defining features columns of the dataset
        self.clean_df = self.movies_df[self.defining_features].fillna('')

        self.user_movie = ''

        self.create_index_col()

    def get_user_movie(self):
        # movies are recommended based on this movie
        self.user_movie = input("Enter your favourite movie: ")
        return self.user_movie

    def get_index(self, movie):
        # getting the index of the closest match (input)
        return self.movies_df[self.movies_df['title'] == movie]['index'].values[0]
    
    def create_index_col(self):
        # every dataset need not have index columns 
        # so if there is no index column in that dataset,
        # we create one, so that movies can be uniquely identified
        if 'index' in self.movies_df.columns:
            return
        self.movies_df['index'] = np.arange(self.movies_df.shape[0])

    def recommend_movies(self, movie, number_of_movies):
        for feature in self.defining_features:
            self.combined_features += self.clean_df[feature]

        # transforming combined_features' textual data to numerical values
        # so that it's easier for the computer to process
        vectorizer = TfidfVectorizer()
        transformed_features = vectorizer.fit_transform(self.combined_features)

        # finding similarities between the movies using the cosine_similarity
        similarity = cosine_similarity(transformed_features)

        # if the movie name entered by the user
        # may not match exactly with the titles we have in our dataset
        # so closest matches can be found using difflib library
        close_matches = difflib.get_close_matches(movie, list(self.movies_df['title']))
        if not close_matches:
            print("no matches found!")
            return
        closest_match = close_matches[0]

        # similar movies to closest match can be known only when we know the index of this movie
        movie_index = self.get_index(closest_match)
        # similar movies are sorted in descending order
        similar_movies = list(enumerate(similarity[movie_index]))
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

        print(f"\nMovies recommended for you: ")
        for i in range(number_of_movies):
            # extracting the title of the movie using it's index
            index = sorted_similar_movies[i][0]
            movie = self.movies_df[self.movies_df['index'] == index]['title'].values[0]
            print(movie)


# based on genres, keywords, tagline, cast and director
rc = RecommenderSystem("movies.csv", ["genres", "keywords", "cast", "director"])
user_movie = rc.get_user_movie()
number_of_movies = 10
rc.recommend_movies(user_movie, number_of_movies)
