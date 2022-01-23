# SimpleClassification

### Overview
A small project to practice Classification on UCI Maternal Health Risk Dataset
Classifier used:
 - LogisticRegression
 - K Nearest Neighbor
 - Support Vector Machine
 
`GridSearchCV` in conjuntion with Pipeline for cross validation and hyperparameter tuning

### Output:
Best model for prediction in current version: 
` KNeighborsClassifier(metric='manhattan', n_neighbors=1), 'lr_classifier__metric': 'manhattan', `

```
precision:  0.8329379451723891  , recall:  0.8327868852459016 , fscore:  0.8328249812915539  , accuracy:  0.8327868852459016
              precision    recall  f1-score   support

           0       0.90      0.89      0.90        82
           1       0.82      0.84      0.83       122
           2       0.79      0.78      0.79       101

    accuracy                           0.83       305
   macro avg       0.84      0.84      0.84       305
weighted avg       0.83      0.83      0.83       305
```

*TODO:*
- Try out more model: Random Forest, Stochastic Gradient Descent (SGD), Decision Tree
- May try to plot ROC curve for multiclass problem
- More EDA graph 
