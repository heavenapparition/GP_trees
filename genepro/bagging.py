import numpy as np
from genepro.scikit import GeneProRegressor

class Bagging:
    def __init__(self, bootstrap_ratio=0.7, with_replacement=True,
                 score=None,
                 use_linear_scaling=True,
                 num_of_trees=10,
                 evo_kwargs=dict(),
                 ):
        self.B = num_of_trees
        self.bootstrap_ratio = bootstrap_ratio
        self.with_replacement = with_replacement
        self.models = [
            GeneProRegressor(score=score, use_linear_scaling=use_linear_scaling, evo_kwargs=evo_kwargs) for _ in range(num_of_trees)
        ]

    def fit(self, X, y):  # <---- Training function
        m, n = X.shape
        # size of mini training set
        size = int(self.bootstrap_ratio * len(X))

        xsamples = np.zeros((self.B, size, n))
        ysamples = np.zeros((self.B, size))

        xsamples_oob = []  # use list because length is not known
        ysamples_oob = []

        # subsamples for each model
        for i in range(self.B):
            oob_idx = []
            idxes = []
            for j in range(size):
                idx = np.random.randint(low=0, high=m - 1)
                if (self.with_replacement):
                    xsamples[i, j, :] = X.iloc[idx]
                    ysamples[i, j] = y.iloc[idx]
                    idxes.append(idx)
                else:
                    if j not in idxes:
                        xsamples[i, j, :] = X[j]
                        ysamples[i, j] = y[j]
                    else:
                        oob_idx.append(j)
            mask = np.zeros((m), dtype=bool)
            mask[idxes] = True
            xsamples_oob.append(X[~mask])
            ysamples_oob.append(y[~mask])
        # fitting each estimator
        oob_score = 0
        for i, model in enumerate(self.models):
            _X = xsamples[i, :]
            _y = ysamples[i, :]
            model.fit(_X, _y)

            # calculating oob score
            _X_test = np.asarray(xsamples_oob[i])
            _y_test = np.asarray(ysamples_oob[i])
            yhat = model.predict(_X_test)
            cur_model_oob_score = model.score(_y_test, yhat)
            oob_score += cur_model_oob_score
            print(f"tree {i} oob_score {cur_model_oob_score}")
        self.avg_oob_score = oob_score / len(self.models)
        print('avg oob score', self.avg_oob_score)
        return self

    def predict(self, X):
        predictions = np.zeros((len(X), 1))
        for i, model in enumerate(self.models):
            yhat = model.predict(X)
            if not i:
                predictions = yhat
            else:
                predictions += yhat
        return predictions