from toolz import curry
import scipy
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import KFold

from .metrics import EPSILON

@curry
def nlpd_rescaled(x, mean, std, target):
    return -(scipy.stats.norm(mean+x[0], EPSILON + std*x[1]).logpdf(target)).mean()


def fit_uncertainty(estimator, X, y, train, val):
    fitted_estimator = estimator.fit(X[train], y[train])
    pred_mean, pred_std = fitted_estimator.predict(X[val])
    opt_res = scipy.optimize.minimize(
        nlpd_rescaled(mean=pred_mean, std=pred_std, target=y[val]),
        x0=[0, 1.0],
        bounds=[(None, None), (0.0, None)],
        jac=False
    )
    return fitted_estimator, opt_res


def predict_uncertainty(estimator, opt_res, X):
    pred_mean, pred_std = estimator.predict(X)
    x = opt_res['x']
    return pred_mean + x[0], pred_std*x[1]


class UncertaintyCalibrator(BaseEstimator, RegressorMixin):

    def __init__(self, estimator, uncertainty_calibrator_cv=None):
        self.estimator = clone(estimator)
        if uncertainty_calibrator_cv is None:
            self.uncertainty_calibrator_cv = KFold(n_splits=5)
        else:
            self.uncertainty_calibrator_cv = uncertainty_calibrator_cv
        self.calibrated_estimators_ = None

    def fit(self, X, y, n_jobs=1, verbose=0):
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
        self.calibrated_estimators_ = parallel(delayed(fit_uncertainty)(clone(self.estimator), X, y, train, val)
                                               for train, val in self.uncertainty_calibrator_cv.split(X, y))
        return self

    def predict(self, X, y=None):
        calibrated_preds = [predict_uncertainty(est, opt_res, X) for est, opt_res in self.calibrated_estimators_]
        calibrated_means, calibrated_stds = zip(*calibrated_preds)
        return np.mean(calibrated_means, axis=0), np.mean(calibrated_stds, axis=0)

    def get_params(self, deep=True):
        params = self.estimator.get_params()
        params['uncertainty_calibrator_cv'] = str(self.uncertainty_calibrator_cv)
        params['estimator'] = type(self.estimator).__name__
        return params

    def set_params(self, **params):
        if 'uncertainty_calibrator_cv' in params:
            self.uncertainty_calibrator_cv = params['uncertainty_calibrator_cv']
        if 'estimator' in params:
            self.estimator = clone(params['estimator'])
        self.estimator.set_params(**params)
        self.calibrated_estimators_ = None
        return self
