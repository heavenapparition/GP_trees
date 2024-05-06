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


class GPGBRegressor:
    def __init__(self,
                 learning_rate=0.1,
                 random_state=0,
                 score=None,
                 use_linear_scaling=True,
                 num_of_trees=10,
                 evo_kwargs=dict(),
                 ):
        self.score = score
        self.evo_kwargs = evo_kwargs
        self.num_of_trees = num_of_trees
        self.use_linear_scaling = use_linear_scaling
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        self.initial_leaf = y.mean()
        predictions = np.zeros(len(y)) + self.initial_leaf

        for _ in tqdm(range(self.num_of_trees)):
            residuals = y - predictions
            tree = GeneProRegressor(score=self.score,
                                    use_linear_scaling=self.use_linear_scaling,
                                    evo_kwargs=self.evo_kwargs
                                    )
            tree.fit(X, residuals)
            predictions += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, samples):
        predictions = np.zeros(len(samples)) + self.initial_leaf

        for i in tqdm(range(self.num_of_trees)):
            predictions += self.learning_rate * self.trees[i].predict(samples)

        return predictions