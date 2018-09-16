import json
from itertools import product
import os
import logging

from bayesian_uncertainty.datasets import make_regression_datasets
from bayesian_uncertainty.evaluation import eval_dataset_model
from bayesian_uncertainty.deep_models import MLPNormal, MLPBaseline, MLPBayesianDropout

logging.getLogger("bayesian_uncertainty").setLevel(logging.INFO)

datasets = make_regression_datasets(make_year=True, make_flight=True)

models = [MLPNormal(),
          MLPBaseline(),
          MLPBayesianDropout()]
    
results = [eval_dataset_model(d, m) for d, m in product(datasets, models)]

results_folder = os.path.join(os.path.dirname(__file__), '../results')

with open(f'{results_folder}/deep_results.json', 'w') as f:
    f.write(json.dumps(results, indent=4))
