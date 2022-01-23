import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score, f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from yellowbrick.classifier import ROCAUC


class Model:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.feature_cols = self.dataframe.iloc[:, :-1]
        self.target_col = self.dataframe.RiskLevel

    def train_test_split(self):
        ss = StratifiedShuffleSplit(n_splits=3, test_size=0.3)
        idx_train, idx_test = next(ss.split(self.feature_cols, self.target_col))
        x_train = self.dataframe.loc[idx_train, self.feature_cols.columns]
        y_train = self.dataframe.loc[idx_train, "RiskLevel"]

        x_test = self.dataframe.loc[idx_test, self.feature_cols.columns]
        y_test = self.dataframe.loc[idx_test, "RiskLevel"]

        return x_train, y_train, x_test, y_test


    def lr_predicting(self, x_train, y_train, x_test):
        #Using pipeline to choose best regularization parameters for regularized LogisticRegression
        pipe_param = [
            {
                'lr_classifier' : [LogisticRegression()],
                'lr_classifier__penalty': ['l1', 'l2'],
                'lr_classifier__C': np.logspace(-4, 4, 20),
                'lr_classifier__dual': [False]
            },
            {
                'lr_classifier': [KNeighborsClassifier()],
                'lr_classifier__metric': ['euclidean', 'manhattan', 'minkowski'],
                'lr_classifier__n_neighbors': range(1, 21, 1)
            },
            {
                'lr_classifier': [SVC()],
                'lr_classifier__C': np.arange(1, 21, 0.1, dtype=float),
                'lr_classifier__kernel': ['poly', 'rbf', 'sigmoid']
            }
        ]

        lr_pipeline = Pipeline([ ('scaler', StandardScaler()), ('lr_classifier', LogisticRegression())])

        grid= GridSearchCV(lr_pipeline, param_grid=pipe_param, cv=5, verbose=True)
        best_grid_clf = grid.fit(x_train, y_train)
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        params = grid.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        print(best_grid_clf.best_params_)

        return best_grid_clf.predict(x_test)

    @staticmethod
    def scoring(y_test, y_prediction):
        precision, recall, fscore, _ = score(y_test, y_prediction, average='weighted')
        accuracy = accuracy_score(y_test, y_prediction)
        # fpt, tpr, thresholds = roc_curve(y_test, y_prediction)
        
        #ROC Curve plotting
        # plt.plot(fpt, tpr, linewidth=3)
        # plt.plot([0,1], [0,1], 'k--')
        # plt.axis([0,1,0,1])
        # plt.show()

        _, ax = plt.subplots(figsize=(12,12))
        ax = sns.heatmap(confusion_matrix(y_test, y_prediction),  annot=True, fmt='d', annot_kws={"size": 40, "weight": "bold"})
        labels = ['False', 'True']
        plt.show()
        print("precision: ", precision, " , recall: ", recall, ", fscore: ", fscore, " , accuracy: ", accuracy)
        print(classification_report(y_test, y_prediction, target_names=['0', '1', '2']))

