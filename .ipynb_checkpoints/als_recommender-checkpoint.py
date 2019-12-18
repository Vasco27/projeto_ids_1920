import random
import pandas as pd
import numpy as np

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler

class ALSRecommender():


    def __init__(self, iterations = 20, latent = 10, alpha_val = 40, regularizer = 0.1):        
        self.iterations = iterations
        self.features = latent
        self.alpha_val = alpha_val
        self.lambda_val = regularizer

    def fit(self, sparse_data):
        """ Implementation of Alternating Least Squares with implicit data. We iteratively
        compute the user (x_u) and item (y_i) vectors using the following formulas:

        x_u = ((Y.T*Y + Y.T*(Cu - I) * Y) + lambda*I)^-1 * (X.T * Cu * p(u))
        y_i = ((X.T*X + X.T*(Ci - I) * X) + lambda*I)^-1 * (Y.T * Ci * p(i))

        Args:
            sparse_data (csr_matrix): Our sparse user-by-item matrix

            alpha_val (int): The rate in which we'll increase our confidence
            in a preference with more interactions.

            iterations (int): How many times we alternate between fixing and 
            updating our user and item vectors

            lambda_val (float): Regularization value

            features (int): How many latent features we want to compute.

        Returns:     
            X (csr_matrix): user vectors of size users-by-features

            Y (csr_matrix): item vectors of size items-by-features
         """

        # Calculate the foncidence for each value in our data
        confidence = sparse_data * self.alpha_val

        # Get the size of user rows and item columns
        user_size, item_size = sparse_data.shape

        # We create the user vectors X of size users-by-features, the item vectors
        # Y of size items-by-features and randomly assign the values.
        X = sparse.csr_matrix(np.random.normal(size = (user_size, self.features)))
        Y = sparse.csr_matrix(np.random.normal(size = (item_size, self.features)))

        #Precompute I and lambda * I
        X_I = sparse.eye(user_size)
        Y_I = sparse.eye(item_size)

        I = sparse.eye(self.features)
        lI = self.lambda_val * I

        # Start main loop. For each iteration we first compute X and then Y
        for i in range(self.iterations):
            print('iteration', i+1, 'of', self.iterations)

            # Precompute Y-transpose-Y and X-transpose-X
            yTy = Y.T.dot(Y)
            xTx = X.T.dot(X)

            # Loop through all users
            for u in range(user_size):

                # Get the user row.
                u_row = confidence[u,:].toarray() 

                # Calculate the binary preference p(u)
                p_u = u_row.copy()
                p_u[p_u != 0] = 1.0

                # Calculate Cu and Cu - I
                CuI = sparse.diags(u_row, [0])
                Cu = CuI + Y_I

                # Put it all together and compute the final formula
                yT_CuI_y = Y.T.dot(CuI).dot(Y)
                yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)
                X[u] = spsolve(yTy + yT_CuI_y + lI, yT_Cu_pu)


            for i in range(item_size):

                # Get the item column and transpose it.
                i_row = confidence[:,i].T.toarray()

                # Calculate the binary preference p(i)
                p_i = i_row.copy()
                p_i[p_i != 0] = 1.0

                # Calculate Ci and Ci - I
                CiI = sparse.diags(i_row, [0])
                Ci = CiI + X_I

                # Put it all together and compute the final formula
                xT_CiI_x = X.T.dot(CiI).dot(X)
                xT_Ci_pi = X.T.dot(Ci).dot(p_i.T)
                Y[i] = spsolve(xTx + xT_CiI_x + lI, xT_Ci_pi)
    
        self.user_vec = X
        self.item_vec = Y
        return X, Y

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
    
    
    
    def recommend(self, user_id, data_sparse, item_lookup, num_items=10):  
        # Get all interactions by the user
        user_interactions = data_sparse[user_id,:].toarray()

        # We don't want to recommend items the user has consumed. So let's
        # set them all to 0 and the unknowns to 1.
        user_interactions = user_interactions.reshape(-1) + 1 #Reshape to turn into 1D array
        user_interactions[user_interactions > 1] = 0

        # This is where we calculate the recommendation by taking the 
        # dot-product of the user vectors with the item vectors.
        rec_vector = self.user_vec[user_id,:].dot(self.item_vec.T).toarray()

        # Let's scale our scores between 0 and 1 to make it all easier to interpret.
        min_max = MinMaxScaler()
        rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
        recommend_vector = user_interactions*rec_vector_scaled

        # Get all the artist indices in order of recommendations (descending) and
        # select only the top "num_items" items. 
        item_idx = np.argsort(recommend_vector)[::-1][:num_items]

        movies = []
        scores = []

        # Loop through our recommended artist indicies and look up the actial artist name
        for idx in item_idx:
            movies.append(item_lookup.title.loc[item_lookup.movie_id == str(idx)].iloc[0])
            scores.append(recommend_vector[idx])

        # Create a new dataframe with recommended artist names and scores
        recommendations = pd.DataFrame({'movie': movies, 'score': scores})

        return recommendations