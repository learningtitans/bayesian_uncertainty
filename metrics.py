import numpy as np
from scipy import stats, optimize
from collections import namedtuple

from sklearn.model_selection import cross_validate, ShuffleSplit, KFold, RepeatedKFold
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from datetime import datetime

Results = namedtuple('Results', 'datetime dataset model shape normal_nll normal_nll_opt rmse auc_rmse auc_rmse_norm')


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
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    cv_metrics = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train = scaler_X.fit_transform(X_train)
        y_train = np.ravel(scaler_y.fit_transform(y_train.reshape(-1, 1)))
        reg.fit(X_train, y_train)
        X_test = scaler_X.transform(X_test)
        pred_mean, pred_std = reg.predict(X_test)
        pred_mean = np.ravel(scaler_y.inverse_transform(pred_mean))
        pred_std *= np.sqrt(scaler_y.var_)
        cv_metrics.append((
            normal_nll(y_test, pred_mean, pred_std),
            normal_nll_opt(y_test, pred_mean, pred_std),
            rmse(y_test, pred_mean),
            auc_rmse(y_test, pred_mean, pred_std),
            auc_rmse_norm(y_test, pred_mean, pred_std)))

    metrics_mean = np.mean(cv_metrics, axis = 0)
    metrics_stderr = stats.sem(cv_metrics, axis = 0)

    r = Results(
        str(datetime.now()),
        d, 
        model.__name__,
        X.shape,
        *zip(metrics_mean, metrics_stderr)
    )
    print(r)
    return r


def normal_nll(actual, pred, std):
    error = np.array(actual) - np.array(pred)
    std[std <= 1e-30] = 1e-30
    return -stats.norm.logpdf(error, loc=0, scale=std).mean()


def normal_nll_opt(actual, pred, std):
    error = np.array(actual) - np.array(pred)
    std[std <= 1e-30] = 1e-30
    func = lambda x: -stats.norm.logpdf(error, loc=x[0], scale=x[1]*std).mean()
    x, f, d = optimize.fmin_l_bfgs_b(func, np.array([0.0, 1.0]), bounds=[(None, None), (0, None)], approx_grad=True)
    return f


def rmse(actual, pred):
    error = np.array(actual) - np.array(pred)
    return np.sqrt((error**2).mean())


def mae(actual, pred):
    error = np.array(actual) - np.array(pred)
    return np.abs(error).mean()


def auc_rmse(actual, pred, std):
    error = np.array(actual) - np.array(pred)
    rmses = []
    data = sorted(zip(std, error), reverse=True)
    for i in range(len(data)):
        _, err = zip(*data[i:])
        rmses.append(np.sqrt((np.array(err)**2).mean()))
    return np.mean(rmses)


def auc_rmse_norm(actual, pred, std):
    base_rmse = rmse(actual, pred)
    error = np.array(actual) - np.array(pred)
    rmses = []
    data = sorted(zip(std, error), reverse=True)
    for i in range(len(data)):
        _, err = zip(*data[i:])
        rmses.append(np.sqrt((np.array(err)**2).mean())/base_rmse)
    return np.trapz(y=rmses, x=np.arange(len(data))/(len(data)-1))


def auc_mae(actual, pred, std):
    error = np.array(actual) - np.array(pred)
    maes = []
    data = sorted(zip(std, error), reverse=True)
    for i in range(len(data)):
        _, err = zip(*data[i:])
        maes.append(np.abs(err).mean())
    return np.mean(maes)