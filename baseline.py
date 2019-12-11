import pandas as pd
import numpy as np

class make_baseline():
    
    def __init__(self, sample, damping_factor = 25):
        self.S = sample
        self.damp_factor = damping_factor
        
    def _user_baseline(self):
        mu = self.S.rating.mean()
        iu = self.S.userId.value_counts()
        bu = self.S.groupby("userId")["rating"].sum().subtract(iu * mu).divide(iu + 25).reset_index()
        bu.columns = ["userId", "bu"]
        return bu
    
    def _movie_baseline(self):
        mu = self.S.rating.mean()
        ui = self.S.movieId.value_counts()
        bu = self._user_baseline()
        bi = self.S.merge(bu, on = "userId", how = "left")
        bi["residual_bi"] = bi["rating"] - bi["bu"] - mu
        bi = bi.groupby("movieId")["residual_bi"].sum().divide(ui + 25).reset_index()
        bi.columns = ["movieId", "bi"]
        return bi
    
    def get_ratings(self):
        bu = self._user_baseline()
        bi = self._movie_baseline()
        bui = self.S.merge(bi, on = "movieId", how = "left")
        bui = bui.merge(bu, on = "userId", how = "left")
        bui["bui"] = bui["rating"] + bui["bu"] + bui["bi"]
        return bui