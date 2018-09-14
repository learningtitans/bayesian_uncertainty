import numpy as np
from scipy import stats, optimize
from collections import namedtuple
import time

from sklearn.model_selection import ShuffleSplit, KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from src.bayesian_uncertainty import datasets

Results = namedtuple('Results', 'datetime dataset model shape train_time test_time normal_nll normal_nll_opt normal_nll_opt2 normal_nll_opt3 normal_nll_opt4 normal_nll_opt5 normal_nll_opt6 rmse auc_rmse auc_rmse_norm')


def eval_dataset_model(dataset, model):    
    try:
        X, y = getattr(datasets, dataset)()
        X = X.values.astype(np.float64)
        y = y.values
    except AttributeError:
        pass
    
    if dataset == 'year':
        cv = ShuffleSplit(1, test_size=0.1)
    elif dataset == 'protein':
        cv = KFold(n_splits=3)
    elif dataset.startswith('make'):
        cv = ShuffleSplit(1, test_size=0.9)
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
        
        start = time.time()
        reg.fit(X_train, y_train)
        train_time = time.time() - start
        
        train_mean, train_std = reg.predict(X_train)
        train_mean = np.ravel(scaler_y.inverse_transform(train_mean))
        train_std *= np.sqrt(scaler_y.var_)
        a, b = optimal_scaling(y_train, train_mean, train_std)  
        c = optimal_scaling2(y_train, train_mean, train_std)
        d = np.mean(y_train - train_mean)
        
        X_test = scaler_X.transform(X_test)
        
        start = time.time()
        pred_mean, pred_std = reg.predict(X_test)
        test_time = time.time() - start
        
        pred_mean = np.ravel(scaler_y.inverse_transform(pred_mean))
        pred_std *= np.sqrt(scaler_y.var_)
        optimal_mean = pred_mean - a
        optimal_std = b*pred_std
        optimal_std2 = c*pred_std
        optimal_mean2 = pred_std - d

        cv_metrics.append((
            train_time,
            test_time,
            normal_nll(y_test, pred_mean, pred_std),
            normal_nll(y_test, optimal_mean, pred_std),
            normal_nll(y_test, pred_mean, optimal_std),
            normal_nll(y_test, optimal_mean, optimal_std),
            normal_nll(y_test, pred_mean, optimal_std2),
            normal_nll(y_test, optimal_mean2, pred_std),
            normal_nll(y_test, optimal_mean2, optimal_std2),
            rmse(y_test, pred_mean),
            auc_rmse(y_test, pred_mean, pred_std),
            auc_rmse_norm(y_test, pred_mean, pred_std)))

    metrics_mean = np.mean(cv_metrics, axis = 0)
    metrics_stderr = stats.sem(cv_metrics, axis = 0)

    r = Results(
        str(datetime.now()),
        dataset, 
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


def optimal_scaling(y_train, train_mean, train_std):
    error = np.array(y_train) - np.array(train_mean)
    train_std[train_std <= 1e-30] = 1e-30
    func = lambda x: -stats.norm.logpdf(error, loc=x[0], scale=np.clip(x[1]*train_std, 1e-60, None)).mean()
    x, f, d = optimize.fmin_l_bfgs_b(func, np.array([0.0, 1.0]), bounds=[(None, None), (1e-30, None)], approx_grad=True, m=25, factr=10.0)
    return x[0], x[1]
    

def optimal_scaling2(y_train, train_mean, train_std):
    error = np.array(y_train) - np.array(train_mean)
    train_std[train_std <= 1e-30] = 1e-30
    func = lambda x: -stats.norm.logpdf(error, loc=0.0, scale=np.exp(x[0])*train_std).mean()
    x, f, d = optimize.fmin_l_bfgs_b(func, np.array([0.0]), bounds=[(None, None)], approx_grad=True, m=25, factr=10.0)
    return np.exp(x[0])


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