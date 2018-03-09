import numpy as np
from scipy import stats, optimize
from collections import namedtuple


Results = namedtuple('Results', 'datetime dataset model shape normal_nll rmse mae auc_rmse auc_mae')


def normal_nll(actual, pred, std):
    error = np.array(actual) - np.array(pred)
    std[std <= 0.0] = 1e-30
    return -stats.norm.logpdf(error, loc=0, scale=std).mean()


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


def auc_mae(actual, pred, std):
    error = np.array(actual) - np.array(pred)
    maes = []
    data = sorted(zip(std, error), reverse=True)
    for i in range(len(data)):
        _, err = zip(*data[i:])
        maes.append(np.abs(err).mean())
    return np.mean(maes)