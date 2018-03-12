from joblib import Parallel, delayed
import numpy as np
import scipy
import pickle
import json
from itertools import product

from datasets import make_datasets
from metrics import eval_dataset_model, Results, normal_nll, rmse, mae, auc_rmse, auc_mae
from shallow_models import LinearRegression, BayesianLinearRegression, RFBaseline, RFUncertainty, GBTQuantile, XGBaseline, XGBLogLikelihood

datasets = make_datasets(year=False)

models = [LinearRegression, BayesianLinearRegression, RFBaseline, RFUncertainty, GBTQuantile, XGBaseline, XGBLogLikelihood]
    
par_results = Parallel(n_jobs=8)(delayed(eval_dataset_model)(d, X, y, m) for (d, (X, y)), m in product(datasets.items(), models))

with open('shallow_results.pkl', 'wb') as f:
    pickle.dump(par_results, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('shallow_results.json', 'w') as f:
    f.write(json.dumps(par_results))
