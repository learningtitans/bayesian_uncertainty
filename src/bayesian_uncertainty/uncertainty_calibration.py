from toolz import curry
import scipy
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.externals.joblib import Parallel, delayed


@curry
def nlpd_rescaled(x, mean, std, target):
    return -(scipy.stats.norm(mean+x[0], std*np.exp(x[1])).logpdf(target)).mean()


def fit_uncertainty(estimator, X, y, train, val):
    fitted_estimator = estimator.fit(X[train], y[train])
    pred_mean, pred_std = fitted_estimator.predict(X[val])
    opt_res = scipy.optimize.minimize(nlpd_rescaled(mean=pred_mean, std=pred_std, target=y[val]), x0=[0, 0], jac=False)
    return fitted_estimator, opt_res


def predict_uncertainty(estimator, opt_res, X):
    pred_mean, pred_std = estimator.predict(X)
    x = opt_res['x']
    return pred_mean + x[0], pred_std*np.exp(x[1])


class UncertaintyCalibrator(BaseEstimator, RegressorMixin):

    def __init__(self, estimator, cv):
        self.estimator = estimator
        self.cv = cv
        self.calibrated_estimators = None

    def fit(self, X, y, n_jobs=-1, verbose=0):
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
        self.calibrated_estimators = parallel(delayed(fit_uncertainty)(clone(self.estimator), X, y, train, val)
                                              for train, val in self.cv.split(X, y))
        return self

    def predict(self, X, y=None):
        calibrated_preds = [predict_uncertainty(est, opt_res, X) for est, opt_res in self.calibrated_estimators]
        calibrated_means, calibrated_stds = zip(*calibrated_preds)
        return np.mean(calibrated_means, axis=0), np.mean(calibrated_stds, axis=0)