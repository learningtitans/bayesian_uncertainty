from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn import linear_model
from sklearn import ensemble
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import scipy.stats


class LinearRegression(BaseEstimator, RegressorMixin):

    def __init__(self, **kwargs):
        self.lr = linear_model.LinearRegression(**kwargs)

    def fit(self, X, y):
        self.lr.fit(X, y)
        errors = y - self.lr.predict(X)
        self.std = np.std(errors)
        return self

    def predict(self, X, y=None):
        pred_mean = self.lr.predict(X)
        pred_std = self.std * np.ones(len(pred_mean))
        return pred_mean, pred_std
    
    
class BayesianLinearRegression(BaseEstimator, RegressorMixin):

    def __init__(self, **kwargs):
        self.blr = linear_model.BayesianRidge(**kwargs)

    def fit(self, X, y):
        self.blr.fit(X, y)
        return self

    def predict(self, X, y=None):
        pred_mean, pred_std = self.blr.predict(X, return_std=True)
        return pred_mean, pred_std
    
    
class GBTQuantile(BaseEstimator, RegressorMixin):

    def __init__(self, **kwargs):
        percentile = scipy.stats.norm.cdf(-1) # One Gaussian std
        self.gbt_lower = ensemble.GradientBoostingRegressor(loss='quantile', alpha=percentile, **kwargs)
        self.gbt_upper = ensemble.GradientBoostingRegressor(loss='quantile', alpha=1-percentile, **kwargs)
        self.gbt_median = ensemble.GradientBoostingRegressor(loss='quantile', alpha=0.5, **kwargs)
        
    def fit(self, X, y):
        self.gbt_lower.fit(X, y)
        self.gbt_upper.fit(X, y)
        self.gbt_median.fit(X, y)
        return self

    def predict(self, X, y=None):
        pred_mean = self.gbt_median.predict(X)
        pred_std = (self.gbt_upper.predict(X) - self.gbt_lower.predict(X))/2
        return pred_mean, pred_std


class RFBaseline(BaseEstimator, RegressorMixin):

    def __init__(self, **kwargs):
        self.rf = ensemble.RandomForestRegressor(**kwargs)
        
    def fit(self, X, y):
        self.rf.fit(X, y)
        errors = y - self.rf.predict(X)
        self.std = np.std(errors)
        return self
    
    def predict(self, X, y=None):
        pred_mean = self.rf.predict(X)
        pred_std = self.std * np.ones(len(pred_mean))
        return pred_mean, pred_std
    
    
class RFUncertainty(BaseEstimator, RegressorMixin):
    """
    Based on: http://blog.datadive.net/prediction-intervals-for-random-forests/
    """
    def __init__(self, **kwargs):
        self.rf = ensemble.RandomForestRegressor(**kwargs)
        
    def fit(self, X, y):
        self.rf.fit(X, y)
        errors = y - self.rf.predict(X)
        self.std = np.std(errors)
        return self
    
    def predict(self, X, y=None):
        pred_mean = self.rf.predict(X)
        percentile = scipy.stats.norm.cdf(-1) # One Gaussian std
        dt_pred = np.vstack([dt.predict(X) for dt in self.rf.estimators_])
        err_down = np.percentile(dt_pred, 100*percentile, axis=0)
        err_up = np.percentile(dt_pred, 100*(1-percentile), axis=0)
        pred_std = (err_up - err_down)/2
        pred_std[pred_std <= 0] = self.std
        return pred_mean, pred_std


class XGBaseline(BaseEstimator, RegressorMixin):

    def __init__(self, **kwargs):
        self.xgb_mean = XGBRegressor(**kwargs)

    def fit(self, X, y):
        self.xgb_mean.fit(X, y)
        errors = y - self.xgb_mean.predict(X)
        self.std = np.std(errors)
        return self

    def predict(self, X, y=None):
        pred_mean = self.xgb_mean.predict(X)
        pred_std = self.std * np.ones(len(pred_mean))
        return pred_mean, pred_std
    
    
def ll_objective(y_true, y_pred):
    err = y_true
    log_var = y_pred
    grad = -1/(2*np.exp(log_var))*(1 - 1/np.exp(log_var)*(err**2))
    hess = 1/np.exp(1.5*log_var) - 2/np.exp(2.5*log_var)*(err**2)
    return -1*grad, -1*hess


class XGBLogLikelihood(BaseEstimator, RegressorMixin):  
    
    def __init__(self, **kwargs):
        self.xgb_mean = XGBRegressor(**kwargs)
        self.xgb_log_var = XGBRegressor(objective=ll_objective, **kwargs)

    def fit(self, X, y):
        self.xgb_mean.fit(X, y)
        errors = y - self.xgb_mean.predict(X)
        self.xgb_log_var.fit(X, errors)        
        return self

    def predict(self, X, y=None):
        pred_mean = self.xgb_mean.predict(X)
        pred_std = np.exp(self.xgb_log_var.predict(X)/2)
        return pred_mean, pred_std


class LGBMUncertainty(BaseEstimator, RegressorMixin):

    def __init__(self, **kwargs):
        self.lgb = LGBMRegressor(**kwargs)

    def fit(self, X, y):
        self.lgb.fit(X, y)
        return self

    def predict(self, X, y=None):
        pred = self.lgb.predict(X, pred_leaf=True)

        ind_pred = []
        for row in pred:
            ind_pred.append([self.lgb.booster_.get_leaf_output(i, j) for i, j in enumerate(row)])
        ind_pred = np.vstack(ind_pred)

        pred_mean = ind_pred.sum(axis=1)
        pred_std = ind_pred.std(axis=1)

        return pred_mean, pred_std