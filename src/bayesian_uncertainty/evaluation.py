import time
import logging.config
import toolz as fp
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
    model_name = type(model).__name__
    dataset_name = dataset.__name__
    logger.info(f"Starting evaluation of model {model_name} on dataset {dataset_name}")

    X, y, splits = dataset()

    if type(X) is pd.DataFrame:
        X = X.values
        y = y.values

    cv_metrics = []

    if calibrate_uncertainty:
        if uncertainty_calibrator_cv is None:
            uncertainty_calibrator_cv = KFold(n_splits=5)
        model = UncertaintyCalibrator(model, uncertainty_calibrator_cv)

    for train_index, test_index in splits:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        start = time.time()
        pred_mean, pred_std = model.predict(X_test)
        test_time = time.time() - start

        cv_metrics.append({
            "train_time": train_time,
            "test_time": test_time,
            "nlpd": nlpd(y_test, pred_mean, pred_std),
            "rmse": rmse(y_test, pred_mean),
            "auc_rmse": auc_rmse(y_test, pred_mean, pred_std),
            "auc_rmse_norm": auc_rmse_norm(y_test, pred_mean, pred_std)
        })

    metrics_df = pd.DataFrame(cv_metrics)

    metrics_mean = metrics_df.mean(axis=0)
    metrics_stderr = metrics_df.sem(axis=0)

    results = {
        "current_time": str(datetime.now()),
        "dataset": dataset_name,
        "model": model_name,
        "shape": X.shape,
        "metrics_df": metrics_df.to_dict(),
        "metrics_mean": metrics_mean.to_dict(),
        "metrics_stderr": metrics_stderr.to_dict()
    }

    logger.info(f"{dataset_name}-{model_name}:\n{metrics_mean}")

    return results
