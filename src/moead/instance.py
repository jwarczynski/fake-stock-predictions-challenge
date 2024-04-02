class Instance:
    def __init__(self, cov_matrix, predictions):
        self.cov_matrix = cov_matrix
        self.predictions = predictions
        self.assets_number = len(predictions)

    def normalize(self):
        predictions_min = self.predictions.min()
        predictions_max = self.predictions.max()
        predictions = (self.predictions - predictions_min) / (predictions_max - predictions_min)

        cov_matrix_min = self.cov_matrix.min()
        cov_matrix_max = self.cov_matrix.max()
        cov_matrix = (self.cov_matrix - cov_matrix_min) / (cov_matrix_max - cov_matrix_min)

        return Instance(cov_matrix, predictions)
