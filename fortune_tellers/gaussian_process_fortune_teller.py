import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV


class GaussianProcessFortuneTeller:
    def __init__(self):
        self.__asset_expected_returns = {}

        periodic_kernel = kernels.ConstantKernel(1.0) * kernels.ExpSineSquared(length_scale=1.0, periodicity=100.0)
        linear_kernel = kernels.ConstantKernel(1.0) * kernels.RBF(length_scale=1.0)
        white_kernel = kernels.WhiteKernel(noise_level=1e-5)

        self.__kernel = periodic_kernel + linear_kernel + white_kernel

    def make_prediction(self, times, prices):
        X = np.array(times).reshape(-1, 1)
        y = np.array(prices).reshape(-1, 1)

        model = self.__get_tuned_model(X, y)

        X_pred = np.array(range(len(X), len(X) + 100)).reshape(-1, 1)
        y_pred, sigma = model.predict(X_pred, return_std=True)

        return y_pred

    def __get_tuned_model(self, X, y):
        param_grid = {
            "kernel__k1__k2__length_scale": (1e-8, 1e2),
            "kernel__k2__k2__length_scale": (1e-8, 1e2),
            "n_restarts_optimizer": [10, 20, 30]
        }

        model = GaussianProcessRegressor(kernel=self.__kernel, n_restarts_optimizer=10)
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv,
                                   scoring='neg_mean_squared_error')
        grid_search.fit(X, y)

        return grid_search.best_estimator_
