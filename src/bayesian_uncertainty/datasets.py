from collections import OrderedDict, namedtuple
import pandas as pd
from sklearn.datasets import load_boston, \
    make_regression as mr, make_friedman1 as mf1, make_friedman2 as mf2, make_friedman3 as mf3


Data = namedtuple('Data', 'X y')


def boston():
    return Data(load_boston()['data'], load_boston()['target'])


def concrete():
    df = pd.read_csv('datasets/concrete_data.csv')
    return Data(df.iloc[:, :-1], df.iloc[:, -1])


def energy():
    df = pd.read_csv('datasets/energy_efficiency.csv')
    return Data(df.iloc[:, :-2], df.iloc[:, -2])


def kin8nm():
    df = pd.read_csv('datasets/kin8nm.csv')
    return Data(df.iloc[:, :-1], df.iloc[:, -1])


def naval():
    df = pd.read_table('datasets/naval.txt', sep='\s+', header=None)
    return Data(df.iloc[:, :-2], df.iloc[:, -2])


def power():
    df = pd.read_csv('datasets/power.csv')
    return Data(df.iloc[:, :-1], df.iloc[:, -1])


def protein():
    df = pd.read_csv('datasets/protein.csv')
    return Data(df.iloc[:, 1:], df.iloc[:, 0])


def wine():
    df = pd.read_csv('datasets/wine.csv', sep=';')
    return Data(df.iloc[:, :-1], df.iloc[:, -1])


def yacht():
    df = pd.read_table('datasets/yacht.txt', sep='\s+', header=None)
    return Data(df.iloc[:, :-1], df.iloc[:, -1])


def year():
    df = pd.read_table('datasets/year.txt', sep=',', header=None)
    return Data(df.iloc[:, 1:], df.iloc[:, 0])


def make_regression():
    return Data(*mr(100000, noise=1.0))


def make_friedman1():
    return Data(*mf1(100000, noise=1.0))


def make_friedman2():
    return Data(*mf2(100000, noise=1.0))


def make_friedman3():
    return Data(*mf3(100000, noise=1.0))


def make_flight():



def make_datasets(year=False, fake=False, flight=False):
    datasets = list()
    datasets.append('boston')
    datasets.append('concrete')
    datasets.append('energy')
    datasets.append('kin8nm')
    datasets.append('naval')
    datasets.append('power')
    datasets.append('protein')
    datasets.append('wine')
    datasets.append('yacht')
    
    if year:
        datasets.append('year')
        
    if fake:
        datasets.append('make_regression')
        datasets.append('make_friedman1')
        datasets.append('make_friedman2')
        datasets.append('make_friedman3')

    return datasets

