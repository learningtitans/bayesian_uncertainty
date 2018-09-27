# Bayesian Uncertainty Baselines for Regression

In this repository we provide baselines to evaluate uncertainty in regression problems.

We provide 11 datasets, 3 uncertainty metrics, 3 deep models, 7 shallow models, and an uncertainty calibration routine.

## Models

1. Multilayer Perceptron
  * Regular (homoskedastic)
  * Bayesian dropout
  * Two outputs: mean and stddev, learning with log-likelihood loss function
2. Extreme gradient boosting
  * Regular (homoskedastic)
  * Tree variance as stddev
  * Two outputs: mean and stddev, learning with log-likelihood loss function
3. Random forest
  * Regular (homoskedastic)
  * Tree variance as stddev
4. Linear (both homoskedastic)
  * Linear regression
  * Bayesian linear regression

## Uncertainty calibration

Uncertainty calibration is a procedure where we calibrate the uncertainty on a validation set in order to maximize the predictive log likelihood (normal distribution):

$calibrated_mean = mean + \alpha$
$calibrated_stddev = stddev * \beta$

Where $\alpha$ and $\beta$ are learned in a validation set. If there are multiple validation sets (e.g. cross-validation), we average the calibrated_mean and calibrated_stddev to make the final predictions.

## Datasets

We provide 10 UCI regression datasets typically used in the bayesian deep learning literature plus one extra large dataset (flight delays).

## Metrics
1. NLPD (negative log predictive distribution) of a normal distribution (sometimes knows as negative log likelihood)
2. RMSE (root mean squared error)
3. Area under curve of the RMSE (each point of the curve is the RMSE with the top X% most uncertain samples removed from the test set)
4. Normalized area under curve of the RMSE (normalized by the RMSE itself)

## Results (work in progress)
1. Shallow models make very strong baselines both in RMSE and NLPD, even compared with the state of the art literature
2. Heteroskedastic variance is almost always more useful than homoskedastic, no matter the method or model

## Install

1. Clone the repo locally
2. Go to its directory
3. Install with:
```bash
pip install -e .
```
4. To run a script: `python scripts/shallow_experiments.py`
