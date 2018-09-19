import time
import logging.config
import pandas as pd
import json
import pkgutil

from sklearn.model_selection import KFold
from datetime import datetime

from .metrics import nlpd, rmse, auc_rmse, auc_rmse_norm
from .uncertainty_calibration import UncertaintyCalibrator

logging.config.dictConfig(json.loads(pkgutil.get_data("bayesian_uncertainty", "resources/logging.json")))
logger = logging.getLogger("bayesian_uncertainty")


def eval_dataset_model(dataset, model, calibrate_uncertainty=False, uncertainty_calibrator_cv=None):
    dataset_name = dataset.__name__
    if type(model) is UncertaintyCalibrator:
        model_name = "UncertaintyCalibrator_%s" % type(model.estimator).__name__
    else:
        model_name = type(model).__name__
    logger.info(f"Starting evaluation of model {model_name} on dataset {dataset_name}")

    start_all = time.time()

    X, y, splits = dataset()

    if type(X) is pd.DataFrame:
        X = X.values
        y = y.values

    if calibrate_uncertainty:
        if uncertainty_calibrator_cv is None:
            uncertainty_calibrator_cv = KFold(n_splits=5)
        model = UncertaintyCalibrator(model, uncertainty_calibrator_cv)

    cv_metrics = []

    for train_index, test_index in splits:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        start_test = time.time()
        pred_mean, pred_std = model.predict(X_test)
        test_time = time.time() - start_test

        cv_metrics.append({
            "train_time": train_time,
            "test_time": test_time,
            "nlpd": nlpd(y_test, pred_mean, pred_std),
            "rmse": rmse(y_test, pred_mean)
            # "auc_rmse": auc_rmse(y_test, pred_mean, pred_std),
            # "auc_rmse_norm": auc_rmse_norm(y_test, pred_mean, pred_std)
        })

    metrics_df = pd.DataFrame(cv_metrics)

    metrics_mean = metrics_df.mean(axis=0)
    metrics_stderr = metrics_df.sem(axis=0)

    all_time = time.time() - start_all

    results = {
        "current_time": str(datetime.now()),
        "n_splits": len(cv_metrics),
        "all_time": all_time,
        "dataset": dataset_name,
        "model": model_name,
        "params": model.get_params(),
        "shape": X.shape,
        "metrics_df": metrics_df.to_dict(),
        "metrics_mean": metrics_mean.to_dict(),
        "metrics_stderr": metrics_stderr.to_dict()
    }

    logger.info(f"{dataset_name}-{model_name} done in {all_time//60} min:\n{metrics_mean}")
    logger.debug(results)

    return results
