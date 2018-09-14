from joblib import Parallel, delayed
import json
from itertools import product

from src.bayesian_uncertainty.datasets import make_datasets
from src.bayesian_uncertainty.metrics import eval_dataset_model
from src.bayesian_uncertainty.shallow_models import LinearRegression, BayesianLinearRegression, RFBaseline, \
    RFUncertainty, GBTQuantile, XGBaseline, XGBLogLikelihood

datasets = make_datasets(year=True, fake=True)

models = [LinearRegression, BayesianLinearRegression, RFBaseline, RFUncertainty,
          GBTQuantile, XGBaseline, XGBLogLikelihood]
    
par_results = Parallel(n_jobs=8)(delayed(eval_dataset_model)(d, m) for d, m in product(datasets, models))

with open('shallow_results.json', 'w') as f:
    f.write(json.dumps(par_results, indent=4))
