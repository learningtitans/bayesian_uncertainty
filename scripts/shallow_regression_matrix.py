from joblib import Parallel, delayed
import json
from itertools import product
import os
import logging

from bayesian_uncertainty.datasets import make_regression_datasets
from bayesian_uncertainty.evaluation import eval_dataset_model
from bayesian_uncertainty.shallow_models import LinearRegression, BayesianLinearRegression, RFBaseline, \
    RFUncertainty, GBTQuantile, XGBaseline, XGBLogLikelihood, LGBMUncertainty

logging.getLogger("bayesian_uncertainty").setLevel(logging.DEBUG)

datasets = make_regression_datasets(make_year=True, make_flight=True)

sklearn_models = [
    LinearRegression(),
    BayesianLinearRegression(),
    RFBaseline(n_estimators=100),
    RFUncertainty(n_estimators=100),
    GBTQuantile(n_estimators=100)
]

gb_models = [
    XGBaseline(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.85),
    XGBLogLikelihood(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.85),
    LGBMUncertainty(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.85)
]

par_results = Parallel(n_jobs=32)(delayed(eval_dataset_model)(d, m) for d, m in product(datasets, sklearn_models))
gb_results = [eval_dataset_model(d, m) for d, m in product(datasets, gb_models)]

results = par_results + gb_results

results_folder = os.path.join(os.path.dirname(__file__), '../results')

with open(f'{results_folder}/shallow_results.json', 'w') as f:
    f.write(json.dumps(par_results, indent=4))
