import json
from itertools import product

from src.bayesian_uncertainty.datasets import make_datasets
from src.bayesian_uncertainty.metrics import eval_dataset_model
from src.bayesian_uncertainty.deep_models import MLPNormal, MLPBaseline, MLPBayesianDropout

datasets = make_datasets(year=True, fake=True)

models = [MLPNormal, MLPBaseline, MLPBayesianDropout]
    
results = [eval_dataset_model(d, m) for d, m in product(datasets, models)]

with open('deep_results.json', 'w') as f:
    f.write(json.dumps(results, indent=4))
