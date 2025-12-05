import numpy as np
import pickle
from abc import ABC, abstractmethod


# --------------------------
# Interfície abstracta
# --------------------------
class Inferencia(ABC):

    @abstractmethod
    def load_model(self, path):
        """Carrega un model des d'un fitxer"""
        pass

    @abstractmethod
    def predict(self, user_id, item_id):
        """Predicció de puntuació per a un usuari i un ítem"""
        pass

    @abstractmethod
    def recommend(self, user_id, top_n=10):
        """Recomanacions top_n per un usuari"""
        pass


# --------------------------
# KNN Predictions
# --------------------------
class KNNPredictions(Inferencia):

    def __init__(self, model=None):
        self.model = model

    def load_model(self, model_path="models/knn_item_model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        return self.model

    def predict(self, user_id, item_id):
        model = self.model

        if item_id not in model.item_index or user_id not in model.user_index:
            return 0.0

        i_idx = model.item_index[item_id]
        u_idx = model.user_index[user_id]
        user_ratings = model.item_user_matrix[:, u_idx]

        if model.use_topk:
            neighbors = model.topk_neighbors[i_idx]
            if neighbors is None:
                return 0.0
            nbr_idx, nbr_sims = neighbors
            if len(nbr_idx) == 0:
                return 0.0
            nbr_ratings = user_ratings[nbr_idx]
            mask = nbr_ratings > 0
            if mask.sum() == 0:
                return 0.0
            sims = nbr_sims[mask]
            ratings = nbr_ratings[mask]
            if sims.sum() == 0:
                return 0.0
            return float(np.dot(sims, ratings) / sims.sum())
        else:
            sims = model.topk_neighbors[i_idx]
            rated_mask = user_ratings > 0
            if rated_mask.sum() == 0:
                return 0.0
            sims_rated = sims[rated_mask]
            ratings_rated = user_ratings[rated_mask]
            if sims_rated.sum() == 0:
                return 0.0
            return float(np.dot(sims_rated, ratings_rated) / sims_rated.sum())

    def recommend(self, user_id, top_n=10):
        model = self.model

        if user_id not in model.user_index:
            return []

        u_idx = model.user_index[user_id]
        user_ratings = model.item_user_matrix[:, u_idx]
        unseen_items = np.where(user_ratings == 0)[0]

        preds = []
        for i in unseen_items:
            # CORRECCIÓ: paràmetres invertits
            pred = self.predict(user_id, model.items[i])
            preds.append((i, pred))

        preds.sort(key=lambda x: x[1], reverse=True)
        top = preds[:top_n]
        return [(model.items[i], score) for i, score in top]


# --------------------------
# SVD Predictions
# --------------------------
class SVDPredictions(Inferencia):

    def __init__(self, model=None):
        self.model = model
        self.users = []
        self.items = []

    def load_model(self, model_path="models/svd_model.pkl"):
        with open(model_path, "rb") as f:
            self.model, self.users, self.items = pickle.load(f)
            self.users = list(self.users)
            self.items = list(self.items)
        return self.model

    def predict(self, user_id, item_id):
        if user_id not in self.users or item_id not in self.items:
            return 0.0

        u_idx = self.users.index(user_id)
        i_idx = self.items.index(item_id)
        return float(np.dot(self.model.P[u_idx], self.model.Q[i_idx]))

    def recommend(self, user_id, top_n=10):
        if user_id not in self.users:
            return []

        u_idx = self.users.index(user_id)
        scores = np.dot(self.model.P[u_idx], self.model.Q.T)

        top_idx = np.argpartition(scores, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [(self.items[i], scores[i]) for i in top_idx]
