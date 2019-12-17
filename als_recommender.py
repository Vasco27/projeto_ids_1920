import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import implicit
    
class ALSRecommender():
    def __init__(self, iterations = 20, latent = 10, alpha = 40, regularizer = 0.1):        
        self.iterations = iterations
        self.latent = latent
        self.alpha = 40
        self.regularizer = 0.1
        self.S = None

    def fit(self, sample):
        #Instead of pivot_table the als from the implicit library expects a sparse matrix
        self.S = sample
        self.S.movieId = self.S.movieId.astype("category").cat.codes
        self.S.userId = self.S.userId.astype("category").cat.codes
        self.movie_user_matrix = sparse.csr_matrix((self.S.rating.astype(float), (self.S.movieId, self.S.userId)))
        self.user_movie_matrix = sparse.csr_matrix((self.S.rating.astype(float), (self.S.userId, self.S.movieId)))
        
        confidence = (self.movie_user_matrix * self.alpha).astype("double")

        #Als model with 10 latent factor, lambda = 0.1 and 20 alternating iterations
        als_model = implicit.als.AlternatingLeastSquares(factors = self.latent, regularization = self.regularizer, iterations = self.iterations)
        als_model.fit(confidence)
        
        self.user_vectors = als_model.user_factors
        self.movie_vectors = als_model.item_factors
        
        return self.user_vectors, self.movie_vectors
    
    def similar_to_movie(self, movie_id, n_similar):
        #scores = V dot V.T[movie] -> Item recommendation
        movie_vec = self.movie_vectors[movie_id].T #será que posso fazer com a movie_user_matrix? assim dava para testar

        movie_norms = np.sqrt((self.movie_vectors * self.movie_vectors).sum(axis = 1))

        scores = np.dot(self.movie_vectors, movie_vec) / movie_norms
        top_10 = np.argpartition(scores, -n_similar)[-n_similar:]
        similar = sorted(zip(top_10, scores[top_10] / movie_norms[n_similar]), key=lambda x: -x[1])
        
        titles = []
        similarity = []
        for i in similar:
            idx, sim = i
            titles.append(self.S.loc[self.S.movieId == idx].title.iloc[0])
            similarity.append(sim)
            
        return pd.DataFrame({"movie_title": titles, "similarity": similarity})
    
    #user_movie_matrix deve ser um parametro (para poder passar uma test sample)
    def recommend_to_user(self, user_id, n_movies, user_movie_matrix):
        user_ratings = user_movie_matrix[user_id,:].toarray().reshape(-1) + 1 #ratings do user escolhido (1d array)
        #ratings de filmes não vistos ficam com rating 1, enquanto os que já foram vistos são postos a 0 para não serem recomendados
        #outra vez ao utilizador
        user_ratings[user_ratings > 1] = 0

        user_vectors = sparse.csr_matrix(self.user_vectors)
        movie_vectors = sparse.csr_matrix(self.movie_vectors)
        
        #formula de cálculo de recomendações para utilizadores
        #Ui dot V.T (produto escalar entre vetor do utilizador i e dos filmes todos transpostos)
        recommendation_vector = np.dot(user_vectors[user_id,:], movie_vectors.T).toarray()

        #remover os filmes já vistos pelo utilizador
        recommend_vector = user_ratings * recommendation_vector.reshape(-1)

        #argsort é ascendente
        #::-1 coloca o valor mais alto no inicio do array, o segundo mais alto de seguida, etc.
        scores_idx = np.argsort(recommend_vector)[::-1][:n_movies] #dez valores mais parecidos

        titles = []
        scores = []
        for i in scores_idx:
            titles.append(self.S.loc[self.S.movieId == i].title.iloc[0])
            scores.append(recommend_vector[i])

        return pd.DataFrame({"movie_title" : titles, "score": scores})