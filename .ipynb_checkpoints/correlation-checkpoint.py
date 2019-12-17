import pandas as pd
import numpy as np

class ratings_correlation():
    def __init__(self, sample, pivot, movie, least_common_ratings = 3, ratings_threshold = 50, corr_threshold = 0.5):
        self.S = sample
        self.pivot = pivot
        self.n_movies = least_common_ratings
        self.ratings_threshold = ratings_threshold
        self.corr_threshold = corr_threshold
        self.movie = movie
        
    ### find_intersections:
    #Encontra os filmes em que os utilizadores votaram, sendo que estes também votaram num certo filme (col) (P(X|Y)).
    #* *df* -> pivot_table
    #* *col* -> Filme para comparar ratings
    def _find_intersections(self):
        return self.pivot.apply(lambda x: self._find_number_of_intersections(x.name))
    
    def _find_number_of_intersections(self, col2):
        col1_unique = self.pivot[self.movie].dropna().reset_index().userId.unique()
        col2_unique = self.pivot[col2].dropna().reset_index().userId.unique()
        return np.intersect1d(col1_unique, col2_unique).size
    
    ### most_relevant:
    #Mostra os filmes em que os utilizadores deram ratings em comum com o *movie* e encontra os que têm mais do que certos utilizadores em comum (least_common_ratings)
    #* *df* -> pivot_table
    #* *movie* -> o filme escolhido para comparar
    #* *least_common_ratings* -> threshold para o número de utilizadores que deram ratings aos dois filmes em comum
    def _most_relevant(self):
        intersections = self._find_intersections()
        return intersections[intersections >= self.n_movies]
    
    ### movies_with_counts:
    #Esta função encontra os filmes que possuem mais do que um certo numero de ratings (*ratings_threshold*)
    def _movies_with_counts(self):
        ratings = self.S.groupby("title")["rating"].mean().reset_index()
        ratings["rating_count"] = self.S.groupby("title")["rating"].count().reset_index().rating
        ratings = ratings.set_index("title").loc[self._most_relevant().index.values.tolist()].reset_index()
        return ratings[ratings.rating_count > self.ratings_threshold]
    
    ### movie_correlations:
    #Faz a correlação entre os ratings de um certo filme (*movie*) e os restantes filmes a comparar (*movies_to_compare*)
    def _movie_correlations(self):
        movie_user_ratings = self.pivot[self.movie]
        return self.pivot[self._movies_with_counts().title].corrwith(movie_user_ratings).sort_values(ascending = False)
    
    ### get_correlations:
    #Função que junta todas as outras por conveniência. Retorna uma série com as correlações mais altas, de modo a encontrar uma recomendação.
    #* *sample* -> amostra dos dataset dos ratings
    #* *pivot* -> a pivot_table necessária à comparação dos ratings
    #* *movie* -> o filme para o qual se quer encontrar uma recomendação
    #* *n_movies_to_compare* -> quantidade de utilizadores que têm ratings em comum com o *movie*, para um filme ser considerado relevante
    #* *ratings_threshold* -> número ratings que um filme de ter para ser considerado relevante
    #* *corr_threshold* -> correlação mínima para um filme ser recomendado
    def get_correlations(self):
        similar_to_movie = self._movie_correlations()
        similar_to_movie.drop(self.movie, inplace = True)
        return similar_to_movie[similar_to_movie > self.corr_threshold]