import numba
import numpy as np
from sklearn import metrics as skmetrics

# metrics = {
#     "binary_classification": [
#         "accuracy",
#         "auc",
#         "f1",
#         "logloss",
#         "precision",
#         "recall",
#     ],
# }

class ClassificationMetrics:
    def __init__(self):
        """
        init class for classification metrics
        """
        self.metrics = {
            "accuracy": self._accuracy,
            "auc": self._auc,
            "f1": self._f1,
            "logloss": self._log_loss,
            "precision": self._precision,
            "recall": self._recall,
        }

    def __call__(self, metric, y_test, y_pred, y_proba):
        if metric not in self.metrics:
            raise Exception("Invalid metric passed")
        if metric == "auc":
            if y_proba is not None:
                return self.metrics[metric](y_test, y_proba[:, 1])
            else:
                return np.nan
        elif metric == "logloss":
            if y_proba is not None:
                return self.metrics[metric](y_test, y_proba[:, 1])
            else:
                return np.nan
        else:
            return self.metrics[metric](y_test, y_pred)

    
    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)


    def _auc(self, y_true, y_pred, fast=False):
        if fast is False:
            return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)
        else:
            return self._fast_auc(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)


    @staticmethod
    @numba.jit
    def _fast_auc(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_true = y_true[np.argsort(y_pred)]
        n_false = 0
        auc = 0
        n = len(y_true)
        for i in range(n):
            y_i = y_true[i]
            n_false += 1 - y_i
            auc += y_i * n_false
        auc /= n_false * (n - n_false)
        return auc

    @staticmethod
    def _log_loss(y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)

    
    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)

    
    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)
