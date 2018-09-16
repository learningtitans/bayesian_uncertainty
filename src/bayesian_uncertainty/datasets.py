import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import ShuffleSplit, KFold, RepeatedKFold
import os

datasets_folder = os.path.join(os.path.dirname(__file__), '../../data/datasets')


def boston():
    X = load_boston()['data']
    y = load_boston()['target']
    cv = RepeatedKFold(n_splits=10, n_repeats=4)
    return X, y, cv.split(X)


def concrete():
    df = pd.read_csv(f'{datasets_folder}/concrete_data.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    cv = RepeatedKFold(n_splits=10, n_repeats=4)
    return X, y, cv.split(X)


def energy():
    df = pd.read_csv(f'{datasets_folder}/energy_efficiency.csv')
    X = df.iloc[:, :-2]
    y = df.iloc[:, -2]
    cv = RepeatedKFold(n_splits=10, n_repeats=4)
    return X, y, cv.split(X)


def kin8nm():
    df = pd.read_csv(f'{datasets_folder}/kin8nm.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    cv = RepeatedKFold(n_splits=10, n_repeats=4)
    return X, y, cv.split(X)

def naval():
    df = pd.read_table(f'{datasets_folder}/naval.txt', sep='\s+', header=None)
    X = df.iloc[:, :-2]
    y = df.iloc[:, -2]
    cv = RepeatedKFold(n_splits=10, n_repeats=4)
    return X, y, cv.split(X)


def power():
    df = pd.read_csv(f'{datasets_folder}/power.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    cv = RepeatedKFold(n_splits=10, n_repeats=4)
    return X, y, cv.split(X)


def wine():
    df = pd.read_csv(f'{datasets_folder}/wine.csv', sep=';')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    cv = RepeatedKFold(n_splits=10, n_repeats=4)
    return X, y, cv.split(X)


def yacht():
    df = pd.read_table(f'{datasets_folder}/yacht.txt', sep='\s+', header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    cv = RepeatedKFold(n_splits=10, n_repeats=4)
    return X, y, cv.split(X)


def protein():
    df = pd.read_csv(f'{datasets_folder}/protein.csv')
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    cv = KFold(n_splits=3)
    return X, y, cv.split(X)


def year():
    df = pd.read_table(f'{datasets_folder}/year.txt', sep=',', header=None)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    cv = ShuffleSplit(1, test_size=0.1)
    return X, y, cv.split(X)


def flight():
    train_target = np.load(f'{datasets_folder}/flights/flights-700k-train-targets.npy').squeeze()
    test_target = np.load(f'{datasets_folder}/flights/flights-700k-test-targets.npy').squeeze()
    train_input = np.load(f'{datasets_folder}/flights/flights-700k-train-inputs.npy')
    test_input = np.load(f'{datasets_folder}/flights/flights-700k-test-inputs.npy')

    X = np.vstack((train_input, test_input))
    y = np.concatenate((train_target, test_target))
    splits = [(np.arange(0, len(train_target)), np.arange(len(train_target), len(y)))]

    return X, y, splits


def make_regression_datasets(make_year=False, make_flight=False):
    datasets = list()
    datasets.append(boston)
    datasets.append(concrete)
    datasets.append(energy)
    datasets.append(kin8nm)
    datasets.append(naval)
    datasets.append(power)
    datasets.append(protein)
    datasets.append(wine)
    datasets.append(yacht)
    
    if make_year:
        datasets.append(year)

    if make_flight:
        datasets.append(flight)

    return datasets

