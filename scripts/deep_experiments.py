import json
from itertools import product
import os
from datetime import datetime

from bayesian_uncertainty.datasets import make_regression_datasets
from bayesian_uncertainty.evaluation import eval_dataset_model
from bayesian_uncertainty.deep_models import MLPNormal, MLPBaseline, MLPBayesianDropout

datasets = make_regression_datasets(make_year=True, make_flight=True)

models = [MLPNormal(),
          MLPBaseline(),
          MLPBayesianDropout()]
    
results = [eval_dataset_model(d, m) for d, m in product(datasets, models)]

results_folder = os.path.join(os.path.dirname(__file__), '../results')
time_now = datetime.now().strftime("%Y_%m_%d_%H_%M")

with open(f'{results_folder}/deep_results{time_now}.json', 'w') as f:
    f.write(json.dumps(results, indent=4))
