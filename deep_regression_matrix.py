import numpy as np
import scipy
import pickle
import json
from itertools import product

from datasets import make_datasets
from metrics import eval_dataset_model, Results, normal_nll, rmse, mae, auc_rmse, auc_mae
from deep_models import MLPNormal, MLPBaseline, MLPBayesianDropout

datasets = make_datasets(year=False)

models = [MLPNormal, MLPBaseline, MLPBayesianDropout]
    
results = [eval_dataset_model(d, X, y, m) for (d, (X, y)), m in product(datasets.items(), models)]

with open('deep_results.pkl', 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('deep_results.json', 'w') as f:
    f.write(json.dumps(results))
