"""Fare predictor: loads the exported MLP weights and runs the network with NumPy only."""
import os
import numpy as np
import features

HERE = os.path.dirname(os.path.abspath(__file__))


class FarePredictor:
    def __init__(self, folder=HERE):
        w = np.load(os.path.join(folder, "mlp_weights.npz"))
        self.n_layers = int(w["n_layers"])
        self.W = [w[f"W{i}"].astype("float64") for i in range(self.n_layers)]
        self.b = [w[f"b{i}"].astype("float64") for i in range(self.n_layers)]
        self.x_mean = w["x_mean"].astype("float64")
        self.x_scale = w["x_scale"].astype("float64")
        self.y_mean = float(w["y_mean"])
        self.y_std = float(w["y_std"])
        self.model_features = [str(s) for s in w["model_features"]]

    def predict(self, raw):
        """raw: DataFrame with pickup_datetime, pickup/dropoff longitude + latitude, passenger_count.
        Returns the predicted fare in USD, one value per row."""
        X = features.make_features(raw)[self.model_features].to_numpy("float64")
        h = (X - self.x_mean) / self.x_scale
        for i in range(self.n_layers):
            h = h @ self.W[i] + self.b[i]
            if i < self.n_layers - 1:
                h = np.maximum(h, 0.0)            # ReLU on every hidden layer, linear output
        return h.ravel() * self.y_std + self.y_mean
