import pandas as pd, seaborn as sns
import matplotlib.pyplot as plt

def eda(dataframe):
    print(dataframe.describe())
    print(dataframe.head())
    print(dataframe.info())

    features = dataframe.iloc[:, :-1]
    target_eda = (dataframe.RiskLevel == "high risk").astype(int)
    # target = dataframe.RiskLevel
    fields = list(dataframe.columns[:-1])

    correlations = dataframe[fields].corrwith(target_eda)
    ax = correlations.plot(kind='bar')
    ax.set(ylim=[-1, 1], ylabel='pearson correlation');
    plt.show()
    # print(dataframe[fields].corrwith(target_eda))


    print("")

    # sns.heatmap(dataframe.corr())
    # plt.show()
    # sns.pairplot(dataframe)
    # plt.show()