from xgboost import XGBRegressor
from sklearn import linear_model
from sklearn import ensemble
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import scipy.stats


class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.lr = linear_model.LinearRegression() 

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
    def __init__(self):
        self.blr = linear_model.BayesianRidge() 

    def fit(self, X, y):
        self.blr.fit(X, y)
        return self

    def predict(self, X, y=None):
        pred_mean, pred_std = self.blr.predict(X, return_std=True)
        return pred_mean, pred_std
    
    
class GBTQuantile(BaseEstimator, RegressorMixin):  
    def __init__(self):
        percentile = scipy.stats.norm.cdf(-1) # One Gaussian std
        self.gbt_lower = ensemble.GradientBoostingRegressor(loss='quantile', alpha=percentile)
        self.gbt_upper = ensemble.GradientBoostingRegressor(loss='quantile', alpha=1-percentile)
        self.gbt_median = ensemble.GradientBoostingRegressor(loss='quantile', alpha=0.5)
        
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
    def __init__(self):
        self.rf = ensemble.RandomForestRegressor(n_estimators=100)
        
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
    def __init__(self):
        self.rf = ensemble.RandomForestRegressor(n_estimators=100)
        
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
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample

    def fit(self, X, y):
        self.xgb_mean = XGBRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth, subsample=self.subsample)
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
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample

    def fit(self, X, y):
        self.xgb_mean = XGBRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth, subsample=self.subsample)
        self.xgb_log_var = XGBRegressor(objective=ll_objective, n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth, subsample=self.subsample)
        self.xgb_mean.fit(X, y)
        errors = y - self.xgb_mean.predict(X)
        self.xgb_log_var.fit(X, errors)        
        return self

    def predict(self, X, y=None):
        pred_mean = self.xgb_mean.predict(X)
        pred_std = np.exp(self.xgb_log_var.predict(X)/2)
        return pred_mean, pred_std