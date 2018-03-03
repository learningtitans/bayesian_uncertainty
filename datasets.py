from collections import OrderedDict, namedtuple
import pandas as pd
import numpy as np
import scipy
from sklearn.datasets import load_boston, make_regression, make_friedman1, make_friedman2, make_friedman3


Data = namedtuple('Data', 'X y')


def make_datasets(year=False, fake=True):
    datasets = OrderedDict()

    datasets['boston'] = Data(load_boston()['data'], load_boston()['target'])

    df = pd.read_csv('datasets/concrete_data.csv')
    datasets['concrete'] =  Data(df.iloc[:, :-1], df.iloc[:, -1])

    df = pd.read_csv('datasets/energy_efficiency.csv')
    datasets['energy'] =  Data(df.iloc[:, :-2], df.iloc[:, -2])

    df = pd.read_csv('datasets/kin8nm.csv')
    datasets['kin8nm'] =  Data(df.iloc[:, :-1], df.iloc[:, -1])

    df = pd.read_table('datasets/naval.txt', sep='\s+', header=None)
    datasets['naval'] =  Data(df.iloc[:, :-2], df.iloc[:, -2])

    df = pd.read_csv('datasets/power.csv')
    datasets['power'] =  Data(df.iloc[:, :-1], df.iloc[:, -1])

    df = pd.read_csv('datasets/protein.csv')
    datasets['protein'] =  Data(df.iloc[:, 1:], df.iloc[:, 0])

    df = pd.read_csv('datasets/wine.csv', sep=';')
    datasets['wine'] =  Data(df.iloc[:, :-1], df.iloc[:, -1])

    df = pd.read_table('datasets/yacht.txt', sep='\s+', header=None)
    datasets['yacht'] =  Data(df.iloc[:, :-1], df.iloc[:, -1])

    if year:
        df = pd.read_table('datasets/year.txt', sep=',', header=None)
        datasets['year'] =  Data(df.iloc[:, 1:], df.iloc[:, 0])

    if fake:
        datasets['make_regression'] = Data(*make_regression(10000, noise=1.0))

        datasets['make_friedman1'] = Data(*make_friedman1(10000, noise=0.1))

        datasets['make_friedman2'] = Data(*make_friedman2(20000, noise=1.0))

        datasets['make_friedman3'] = Data(*make_friedman3(30000, noise=10.0))
    
    return datasets