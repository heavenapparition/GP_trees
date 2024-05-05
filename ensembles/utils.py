import numpy as np


class Bootstrapper:
    def __init__(self):
        pass

    @staticmethod
    def get_generator(X: np.ndarray, y: np.ndarray, n_bootstrap_samples: int=10):
        """
        This method is working with ndarrays only
        Parameters
        ----------
        X: array of features
        y: target
        n_bootstrap_samples: number of samples to generate

        Returns
        -------
        generator of samples of similar distribution as X, y as a splitted train/test using OOB

        """
        assert len(X) == len(y)
        features_indexes = np.arange(len(X[0]))
        total_x_indexes = set(np.arange(len(X)))
        arrays_of_indexes = [np.int16(np.random.choice(y, size=len(y), replace=True)) for _ in
                             range(n_bootstrap_samples)]
        for array in arrays_of_indexes:
            features_subspace = sorted(np.int16(
                np.random.choice(features_indexes, size=np.random.randint(low=1, high=len(X[0])), replace=False)))
            oob_indexes = np.array(list(total_x_indexes.difference(set(array))))
            yield {"sample_indexes": array, "feature_indexes": features_subspace, "oob_indexes": oob_indexes}

    @staticmethod
    def get_x_y_by_indexes(dict_of_indexes, X, y=None):
        sample_indexes = dict_of_indexes['sample_indexes']
        feature_indexes = dict_of_indexes['feature_indexes']
        oob_indexes = dict_of_indexes['oob_indexes']
        if y is not None:
            x_train, x_test = X[sample_indexes, :], X[oob_indexes, :]
            y_train, y_test = y[sample_indexes], y[oob_indexes]
            return x_train[:, feature_indexes], y_train, x_test[:, feature_indexes], y_test
        return X[:, feature_indexes]