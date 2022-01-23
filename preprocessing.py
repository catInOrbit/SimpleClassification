import  pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


class DataPreprocessing():
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def encoding(self):
        le = LabelEncoder()
        self.dataframe.RiskLevel = le.fit_transform(self.dataframe.RiskLevel)
        return self.dataframe

    def eda(self):
        print(self.dataframe.describe())
        print(self.dataframe.info())

        sns.heatmap(self.dataframe.corr())
        plt.show()

        sns.pairplot(self.dataframe)
        plt.show()
