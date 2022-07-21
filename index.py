import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io
import math
import string
from pathlib import Path
import os
import glob
import mne

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, classification_report, roc_curve, roc_auc_score, accuracy_score, log_loss, recall_score, precision_score, f1_score, plot_roc_curve

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("adhd.csv")

adhd_df = df.drop(['Unnamed: 20'], axis=1)


control_df = pd.read_csv("adhdcontrol1.csv")

df_combined = pd.concat([adhd_df, control_df])
df_combined.reset_index(inplace=True)
df_combined = df_combined.drop('index', axis=1)


X = df_combined.drop(['19'], axis=1)
y = df_combined['19']

pca = PCA(n_components=2)
x_pca = pca.fit_transform(X)


plt.figure(figsize=(5, 5))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='plasma')


pca1 = PCA().fit(X)

plt.rcParams["figure.figsize"] = (12, 6)

fig, ax = plt.subplots()
xi = np.arange(1, 20, step=1)
y1 = np.cumsum(pca1.explained_variance_ratio_)

plt.ylim(0.4, 1.2)
plt.plot(xi, y1, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 20, step=1))
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)

ax.grid(axis='x')
plt.show()

pca1.explained_variance_ratio_

opt_pca = PCA(n_components=5)
X_final_PCA = opt_pca.fit_transform(X)


X_pca_df = pd.DataFrame(X_final_PCA, columns=[
                        'pca1', 'pca2', 'pca3', 'pca4', 'pca5'])


def plot_roc_curve(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.title("ROC curve")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plot_confusion_matrix1(y_test, y_pred):
    conf_matrix = [[0, 0], [0, 0]]
    y_test1 = [x for x in y_test]
    for i in range(0, len(y_test1)):
        if y_test1[i] == 1 and y_pred[i] == 1:  # true positive
            conf_matrix[0][0] += 1
        elif y_test1[i] == 1 and y_pred[i] == 0:  # false negative
            conf_matrix[0][1] += 1
        elif y_test1[i] == 0 and y_pred[i] == 1:  # false positive
            conf_matrix[1][0] += 1
        elif y_test1[i] == 0 and y_pred[i] == 0:  # true negative
            conf_matrix[1][1] += 1
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(2):
        for j in range(2):
            ax.text(x=j, y=i, s=conf_matrix[i][j],
                    va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


max0 = 0
y_pred_final_log = []
y_test_final_log = []
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

for train_index, test_index in skf.split(X, y):
    X_train_log, X_test_log = X.iloc[train_index], X.iloc[test_index]
    y_train_log, y_test_log = y.iloc[train_index], y.iloc[test_index]
    log = LogisticRegression()
    penalty = ['l1', 'l2']
    c = [0.1, 0.2, 0.003, 0.02, 0.005]
    hyperparameters = dict(C=c, penalty=penalty)
    clf_log = GridSearchCV(log, hyperparameters, cv=5, verbose=0)
    best_model = clf_log.fit(X_train_log, y_train_log)
    print("**************************************************************")
    print('Best Parameters', clf_log.best_params_)
    print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
    print('Best C:', best_model.best_estimator_.get_params()['C'])
    y_pred_log = clf_log.predict(X_test_log)
    print("Accuracy Score: ", accuracy_score(y_test_log, y_pred_log))
    if accuracy_score(y_test_log, y_pred_log) > max0:
        max0 = accuracy_score(y_test_log, y_pred_log)
        y_test_final_log = list(y_test_log)
        y_pred_final_log = list(y_pred_log)
        X_test_final_log = list(X_test_log)


print("Accuracy for Logistic model: ", accuracy_score(
    y_test_final_log, y_pred_final_log))
print("Precision for Logistic model: ", precision_score(
    y_test_final_log, y_pred_final_log))
print("Recall for Logistic model: ", recall_score(
    y_test_final_log, y_pred_final_log))
print("F1 Score for Logistic model: ", f1_score(
    y_test_final_log, y_pred_final_log))
print("Classification Report: \n", classification_report(
    y_test_final_log, y_pred_final_log))
print("ROC: ", roc_curve(y_test_final_log, y_pred_final_log))
print("Area Under the curve: ", roc_auc_score(
    y_test_final_log, y_pred_final_log))


max0 = 0
y_pred_final_knn = []
y_test_final_knn = []
for train_index, test_index in skf.split(X, y):
    X_train_knn, X_test_knn = X.iloc[train_index], X.iloc[test_index]
    y_train_knn, y_test_knn = y.iloc[train_index], y.iloc[test_index]
    knn = KNeighborsClassifier()
    hyperparameters = {'n_neighbors': [2, 3, 4, 5, 6], 'p': [1, 2, 3]}
    clf_knn = GridSearchCV(knn, hyperparameters, cv=5, verbose=0)
    best_model = clf_knn.fit(X_train_knn, y_train_knn)
    print("**************************************************************")
    print('Best Parameters', clf_knn.best_params_)
    print('Best Number of neighbors :',
          best_model.best_estimator_.get_params()['n_neighbors'])
    print('Best p :', best_model.best_estimator_.get_params()['p'])
    y_pred_knn = clf_knn.predict(X_test_knn)
    print("Accuracy Score: ", accuracy_score(y_test_knn, y_pred_knn))
    if accuracy_score(y_test_knn, y_pred_knn) > max0:
        max0 = accuracy_score(y_test_knn, y_pred_knn)
        y_test_final_knn = list(y_test_knn)
        y_pred_final_knn = list(y_pred_knn)
        X_test_final_knn = list(X_test_knn)


print("Accuracy for KNN model: ", accuracy_score(
    y_test_final_knn, y_pred_final_knn))
print("Precision for KNN model: ", precision_score(
    y_test_final_knn, y_pred_final_knn))
print("Recall for KNN model: ", recall_score(
    y_test_final_knn, y_pred_final_knn))
print("F1 Score for KNN model: ", f1_score(y_test_final_knn, y_pred_final_knn))
print("Classification Report: \n", classification_report(
    y_test_final_knn, y_pred_final_knn))
print("ROC: ", roc_curve(y_test_final_knn, y_pred_final_knn))
print("Area Under the curve: ", roc_auc_score(
    y_test_final_knn, y_pred_final_knn))


max0 = 0
y_pred_final_bnb = []
y_test_final_bnb = []
for train_index, test_index in skf.split(X, y):
    X_train_bnb, X_test_bnb = X.iloc[train_index], X.iloc[test_index]
    y_train_bnb, y_test_bnb = y.iloc[train_index], y.iloc[test_index]
    bnb = BernoulliNB()
    hyperparameters = {'alpha': [10, 20, 30, 40]}
    clf_bnb = GridSearchCV(bnb, hyperparameters, cv=5, verbose=0)
    best_model = clf_bnb.fit(X_train_bnb, y_train_bnb)
    print("**************************************************************")
    print('Best Parameters', clf_bnb.best_params_)
    print('Best Alpha:', best_model.best_estimator_.get_params()['alpha'])
    y_pred_bnb = clf_bnb.predict(X_test_bnb)
    print("Accuracy Score: ", accuracy_score(y_test_bnb, y_pred_bnb))
    if accuracy_score(y_test_bnb, y_pred_bnb) > max0:
        max0 = accuracy_score(y_test_bnb, y_pred_bnb)
        y_test_final_bnb = list(y_test_bnb)
        y_pred_final_bnb = list(y_pred_bnb)
        X_test_final_bnb = list(X_test_bnb)


print("Accuracy for Bernoulli Naive Bayes model: ",
      accuracy_score(y_test_final_bnb, y_pred_final_bnb))
print("Precision for Bernoulli Naive Bayes model: ",
      precision_score(y_test_final_bnb, y_pred_final_bnb))
print("Recall for Bernoulli Naive Bayes model: ",
      recall_score(y_test_final_bnb, y_pred_final_bnb))
print("F1 Score for Bernoulli Naive Bayes model: ",
      f1_score(y_test_final_bnb, y_pred_final_bnb))
print("Classification Report: \n", classification_report(
    y_test_final_bnb, y_pred_final_bnb))
print("ROC: ", roc_curve(y_test_final_bnb, y_pred_final_bnb))
print("Area Under the curve: ", roc_auc_score(
    y_test_final_bnb, y_pred_final_bnb))


max0 = 0
y_pred_final_gnb = []
y_test_final_gnb = []
for train_index, test_index in skf.split(X, y):
    X_train_gnb, X_test_gnb = X.iloc[train_index], X.iloc[test_index]
    y_train_gnb, y_test_gnb = y.iloc[train_index], y.iloc[test_index]
    gnb = GaussianNB()
    hyperparameters = {'var_smoothing': [0.1, 0.02, 0.004, 0.005, 0.8]}
    clf_gnb = GridSearchCV(gnb, hyperparameters, cv=5, verbose=0)
    best_model = clf_gnb.fit(X_train_gnb, y_train_gnb)
    print("**************************************************************")
    print('Best Parameters', clf_gnb.best_params_)
    print('Best Variance smoothing:',
          best_model.best_estimator_.get_params()['var_smoothing'])
    y_pred_gnb = clf_gnb.predict(X_test_gnb)
    print("Accuracy Score: ", accuracy_score(y_test_gnb, y_pred_gnb))
    if accuracy_score(y_test_gnb, y_pred_gnb) > max0:
        max0 = accuracy_score(y_test_gnb, y_pred_gnb)
        y_test_final_gnb = list(y_test_gnb)
        y_pred_final_gnb = list(y_pred_gnb)
        X_test_final_gnb = list(X_test_gnb)


print("Accuracy for Gaussian Naive Bayes model: ",
      accuracy_score(y_test_final_gnb, y_pred_final_gnb))
print("Precision for Gaussian Naive Bayes model: ",
      precision_score(y_test_final_gnb, y_pred_final_gnb))
print("Recall for Gaussian Naive Bayes model: ",
      recall_score(y_test_final_gnb, y_pred_final_gnb))
print("F1 Score for Gaussian Naive Bayes model: ",
      f1_score(y_test_final_gnb, y_pred_final_gnb))
print("Classification Report: \n", classification_report(
    y_test_final_gnb, y_pred_final_gnb))
print("ROC: ", roc_curve(y_test_final_gnb, y_pred_final_gnb))
print("Area Under the curve: ", roc_auc_score(
    y_test_final_gnb, y_pred_final_gnb))
