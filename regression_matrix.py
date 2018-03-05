from sklearn.model_selection import cross_validate, ShuffleSplit, KFold, RepeatedKFold
from sklearn.metrics import make_scorer
from datetime import datetime
from joblib import Parallel, delayed
from collections import namedtuple
import numpy as np
import scipy
import pickle
import json
from itertools import product

from datasets import make_datasets
from metrics import normal_nll, rmse, mae, auc_rmse, auc_mae
from shallow_models import LinearRegression, BayesianLinearRegression, RFBaseline, RFUncertainty, GBTQuantile, XGBaseline, XGBLogLikelihood

datasets = make_datasets(year=False)

models = [LinearRegression, BayesianLinearRegression, RFBaseline, RFUncertainty, GBTQuantile, XGBaseline, XGBLogLikelihood]

Results = namedtuple('Results', 'datetime dataset model shape cv_metrics normal_nll rmse mae auc_rmse auc_mae')


def eval_dataset_model(d, X, y, model):    
    try:
        X = X.values
        y = y.values
    except AttributeError:
        pass
    
    if d == 'year':
        cv = ShuffleSplit(1, test_size=0.1)
    elif d == 'protein':
        cv = KFold(n_splits=3)
    elif d.startswith('make'):
        cv = KFold(n_splits=5)#ShuffleSplit(1, test_size=0.8)
    else:
        cv = RepeatedKFold(n_splits=10, n_repeats=4)
    
    reg = model()
    cv_metrics = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg.fit(X_train, y_train)
        pred_mean, pred_std = reg.predict(X_test)
        cv_metrics.append((
            normal_nll(y_test, pred_mean, pred_std),
            rmse(y_test, pred_mean),
            mae(y_test, pred_mean),
            auc_rmse(y_test, pred_mean, pred_std),
            auc_mae(y_test, pred_mean, pred_std)))

    metrics_mean = np.mean(cv_metrics, axis = 0)
    metrics_stderr = scipy.stats.sem(cv_metrics, axis = 0)

    r = Results(
        str(datetime.now()),
        d, 
        model.__name__,
        X.shape,
        cv_metrics,
        *zip(metrics_mean, metrics_stderr)
    )
    print(r)
    return r

    
par_results = Parallel(n_jobs=16)(delayed(eval_dataset_model)(d, X, y, m) for (d, (X, y)), m in product(datasets.items(), models))

with open('par_results.pkl', 'wb') as f:
    pickle.dump(par_results, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('par_results.json', 'w') as f:
    f.write(json.dumps(par_results))
