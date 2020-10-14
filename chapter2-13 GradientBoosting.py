# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 00:06:51 2020

@author: tikut
"""
import mglearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}" .format(gbrt.score(X_train,y_train)))
print("Accuracy on test set: {:.3f}" .format(gbrt.score(X_test,y_test)))


#深さ下げる
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print("---depth=1---")
print("Accuracy on training set: {:.3f}" .format(gbrt.score(X_train,y_train)))
print("Accuracy on test set: {:.3f}" .format(gbrt.score(X_test,y_test)))

#学習率下げる
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
print("---learning rate=0.01---")
print("Accuracy on training set: {:.3f}" .format(gbrt.score(X_train,y_train)))
print("Accuracy on test set: {:.3f}" .format(gbrt.score(X_test,y_test)))

#重要度

gbrt = GradientBoostingClassifier(random_state=0,max_depth=1)
gbrt.fit(X_train, y_train)


def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features),model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features),cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
plot_feature_importances_cancer(gbrt)