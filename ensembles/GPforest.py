import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from ensembles.utils import Bootstrapper
from genepro.evo import Evolution
from genepro.node_impl import *
from genepro.util import compute_linear_scaling
from genepro.scikit import GeneProRegressor
from tqdm import tqdm


class GPForestRegressor(BaseEstimator):
    def __init__(self,
                 score=None,
                 use_linear_scaling=True,
                 num_of_trees=10,
                 evo_kwargs=dict(),
                 ):
        self.num_of_trees = num_of_trees
        self.use_linear_scaling = use_linear_scaling
        # initing all the estimators of an ensemble
        self.estimators = \
            [
                GeneProRegressor(score=score, use_linear_scaling=use_linear_scaling, evo_kwargs=evo_kwargs)
                for _ in range(self.num_of_trees)
            ]
        self.estimator_indexes = []
        # TODO: Сделать обработку параметров дерева

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        bootstrap_gen = Bootstrapper.get_generator(X, y, self.num_of_trees)
        for idx, estimator in tqdm(enumerate(self.estimators)):
            estimator_indexes = next(bootstrap_gen)
            self.estimator_indexes.append(estimator_indexes)
            X_train, y_train, _, _ = Bootstrapper.get_x_y_by_indexes(estimator_indexes, X, y)
            # estimator.set_X_(X_train)
            # estimator.set_y_(y_train)
            # estimator.set_X_(X_train), estimator.y_ = X_train, y_train
            estimator.fit(X_train, y_train)

    def predict(self, X, best_ever=False):
        X = check_array(X)
        predicted = None
        for idx, estimator in tqdm(enumerate(self.estimators)):
            X_transformed = Bootstrapper.get_x_y_by_indexes(self.estimator_indexes[idx], X, y=None)
            if not idx:
                predicted = estimator.predict(X_transformed, best_ever)
            else:
                predicted += estimator.predict(X_transformed, best_ever)
        return predicted / len(self.estimators)


