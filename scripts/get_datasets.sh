#!/usr/bin/env bash

# Small data UCI datasets
unzip ../data/datasets.zip -d ../data/

# Year dataset
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip -P ../data
unzip ../data/YearPredictionMSD.txt.zip -d ../data/
mv ../data/YearPredictionMSD.txt ../data/datasets/year.txt

# Flight delay dataset
mkdir ../data/datasets/flights
unzip ../data/flight-delay-dataset.zip -d ../data/datasets/flights
