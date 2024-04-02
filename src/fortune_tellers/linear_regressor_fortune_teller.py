import numpy as np
from sklearn.linear_model import LinearRegression


class LinearRegressorFortuneTeller:
    def __init__(self, model):
        self.model = model

    def make_prediction(self, times, prices):
        X = np.array(times).reshape(-1, 1)
        y = np.array(prices)
        model = LinearRegression()
        model.fit(X, y)

        future_times = np.arange(X.size, X.size + 100).reshape(-1, 1)
        return model.predict(future_times)
