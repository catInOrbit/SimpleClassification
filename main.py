import pandas as pd, numpy as np, matplotlib.pyplot as plt, os, sys, seaborn as sns
from preprocessing import DataPreprocessing
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from model import Model
from eda import eda

dataframe = pd.read_csv("Maternal Health Risk Data Set.csv")
eda(dataframe)
print(dataframe.RiskLevel.unique())

dp = DataPreprocessing(dataframe)

encoded_df = dp.encoding()
print(encoded_df.head())
print(encoded_df.RiskLevel.unique())

model = Model(encoded_df)
x_train, y_train, x_test, y_test = model.train_test_split()
predictions = model.lr_predicting(x_train, y_train, x_test)
print(predictions)

model.scoring(y_test, predictions)
# dp.EDA()