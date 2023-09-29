# -*- coding: utf-8 -*-

#Importing Necessary Libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.metrics import f1_score

dataset = pd.read_csv('/content/training_sample.csv')

dataset.head()

"""*   Data in all the columns are Nominal data

#EDA - Exploratory Data Anlaysis

## Understanding data
"""

dataset.columns

dataset.describe()

"""The data points are binary encoded.What this means is that every data point is encoded as a sequence of 1’s and 0’s, where each 1 indicates that a certain feature is present and each 0 indicates that the same feature does not exist.


For instance, after the basket icon, when the customer logged into or he has items in his wish list and are stored in the associated columns; then these data points becomes 1 otherwise it’s 0.
"""

has_blank = dataset.isnull().values.any()
print(has_blank)

dataset.info()

dataset.shape

"""#Incorporation Visualizations

Leveraging Heatmaps, it would help me understand relationships between multiple categorical variables.
"""

import seaborn as sns
corr = dataset.corr()
plt.figure(figsize=(16,10))
sns.heatmap(corr, annot=True,fmt='.2f')
plt.show()

dataset.corr()['ordered'] # Finding the correlation of ordered variable to other variables.

"""Finding coorelation with respect to the target variable "orderded"

*   A positive number represents a positive correlation (number closer to 1 showesa  stronger correlation)
*   A negative represents a negative coorelation, whih means that variabl is not contributing to prediction of the target variable.


"""

g = sns.pairplot(dataset, vars=['basket_icon_click', 'basket_add_list', 'basket_add_detail', 'sort_by'])
plt.show()

sns.pairplot(dataset, vars=['image_picker', 'account_page_click', 'promo_banner_click',
       'detail_wishlist_add', 'list_size_dropdown', 'closed_minibasket_click'])
plt.show()

"""###Finalizing Predictors and Target Variables

Droping irrelevant columns
"""

predictors = dataset.drop(['ordered','UserID','device_mobile'], axis=1)
targets = dataset.ordered

print(predictors.columns)

"""## Doing a Train and test Split"""

x_train, x_test, y_train, y_test = train_test_split(predictors, targets, test_size = 0.2)

print("Predictig - training :", x_train.shape, "Predictor - testing: ",x_test.shape)

"""## Training the model using Naive Bayes"""

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier = classifier.fit(x_train, y_train)

y_train.head()

"""##confusion matrix                   
                  [[True Positives, False Positives],
                  [False Negatives, True negatives]]
"""

predictions = classifier.predict(x_test)
#Analysing Accuracy of Prections using Confusion Matrix
sklearn.metrics.confusion_matrix(y_test,predictions)

sklearn.metrics.accuracy_score(y_test, predictions)

f1 = f1_score(y_test, predictions)
print(f1)

"""**Calculating Performance of the model using KFold**"""

from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score, mean_absolute_error

X = predictors
Y = dataset["ordered"]

k = 40
mae_score = [] #mean absolute error
acc_train_scores = []
acc_test_scores = []

kf = KFold(n_splits = 40, shuffle=True, random_state=22)
model = GaussianNB()

for train_index, test_index in kf.split(X):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

  #training the model on the training fold
  model.fit(X_train, Y_train)

  #Evaluating the model on the test fold
  acc_train = r2_score(Y_train, model.predict(X_train))
  acc_test = r2_score(Y_test, model.predict(X_test))
  mae = mean_absolute_error(Y_test, model.predict(X_test))
  print(f"Accuracy Training = {acc_train}, Accuracy Testing = {acc_test}, MAE = {mae}")

  #Calculating the numpy absolute value
  acc_train_scores.append(np.abs(acc_train))
  mae_score.append(np.abs(mae))
  acc_test_scores.append(np.abs(acc_test))

#Calculating average Mean Absolute Error
average_mae = sum(mae_score) / len(mae_score)
print(" ")
print(f"across {k} folds :- \n Average MAE; {average_mae}",
      f"Average training Accuracy: {np.mean(acc_train_scores)}",
      f"Average testing Accuracy: {np.mean(acc_test_scores)}")

"""## **Predicting the behaviour of our new customers**"""

test_dataset = pd.read_csv('/content/testing_sample.csv')

test_dataset.info()

test_dataset.describe()

"""x_old = predictors
y_old = dataset.ordered # or you can use dataset["ordered"]"""

"""#training the model again on old dataset, previously we dropped USERID but now we need it
new_classifier = GaussianNB()
new_classifier.fit(x_old, y_old)"""
count = 0
# New Sample of Predictors and Targets from test_dataset
new_predictors = test_dataset.drop(["ordered","device_mobile", "UserID"], axis=1)
target_pred_proba = classifier.predict_proba(new_predictors)

# Set a threshold
threshold = 0.7

# Identify customers with a higher chance of buying the product
customers_with_high_propensity = []
for i in range(len(target_pred_proba)):
  count += 1
  if target_pred_proba[i][1] >= threshold:
    print(f"Customer with user Id : {test_dataset.UserID[i]} is likely to buy the product")
  else:
    print(f"Customer with user Id : {test_dataset.UserID[i]} Won't buy")
  print(count)
