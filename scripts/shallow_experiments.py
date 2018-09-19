from joblib import Parallel, delayed
import json
from itertools import product
import os
from datetime import datetime

from bayesian_uncertainty.datasets import make_regression_datasets
from bayesian_uncertainty.evaluation import eval_dataset_model
from bayesian_uncertainty.shallow_models import LinearRegression, BayesianLinearRegression, RFBaseline, \
    RFUncertainty, XGBaseline, XGBLogLikelihood, LGBMUncertainty
from bayesian_uncertainty.uncertainty_calibration import UncertaintyCalibrator

datasets = reversed(make_regression_datasets(make_year=True, make_flight=True))

base_models = list(reversed([
    LinearRegression(),
    BayesianLinearRegression(),
    RFBaseline(n_estimators=250, max_depth=7, n_jobs=16),
    RFUncertainty(n_estimators=250, max_depth=7, n_jobs=16),
    XGBaseline(n_estimators=250, max_depth=4, learning_rate=0.1, subsample=0.85, nthread=8),
    XGBLogLikelihood(n_estimators=250, max_depth=4, learning_rate=0.1, subsample=0.85, nthread=8),
    LGBMUncertainty(n_estimators=250, max_depth=4, learning_rate=0.1, subsample=0.85, num_threads=8, verbose=-1)
]))

uncertainty_calibrated_models = [UncertaintyCalibrator(m) for m in base_models]

models = uncertainty_calibrated_models + base_models

results = Parallel(n_jobs=8)(delayed(eval_dataset_model)(d, m) for d, m in product(datasets, models))

results_folder = os.path.join(os.path.dirname(__file__), '../results')
time_now = datetime.now().strftime("%Y_%m_%d_%H_%M")

with open(f'{results_folder}/shallow_results_{time_now}.json', 'w') as f:
    f.write(json.dumps(results, indent=4))
