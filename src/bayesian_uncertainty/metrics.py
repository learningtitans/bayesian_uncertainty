import numpy as np
from scipy import stats

EPSILON = 1e-30


def nlpd(actual, pred_mean, pred_std):
    pred_std[pred_std < EPSILON] = EPSILON
    return -stats.norm(pred_mean, pred_std).logpdf(actual).mean()


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