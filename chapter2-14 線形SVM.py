# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:05:05 2020

@author: tikut
"""
import mglearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs
X,y = make_blobs(centers=4, random_state=8)
y= y%2

mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


#線形SVM
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X,y)

mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#2番目の特徴量の2乗を追加
X_new = np.hstack([X, X[:, 1:] **2])

from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
#3Dで可視化
ax=Axes3D(figure, elev=-152, azim=-26)
#y==0 の点をプロットしてからy==1の点をプロット
mask = y == 0
ax.scatter(X_new[mask,0], X_new[mask,1], X_new[mask,2], c='b',
           cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask,0], X_new[~mask,1], X_new[~mask,2], c='r',marker='^',
           cmap=mglearn.cm2, s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 **2")

#3Dに線形SVM
linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
figure = plt.figure()
ax= Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:,0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:,1].max() + 2, 50)

XX,YY = np.meshgrid(xx,yy)
ZZ = (coef[0] * XX +coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX,YY,ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask,0], X_new[mask,1], X_new[mask,2], c='b',
           cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask,0], X_new[~mask,1], X_new[~mask,2], c='r',marker='^',
           cmap=mglearn.cm2, s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 **2")

#2つの特徴量の関数として表示
figure= plt.figure()
ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(),ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape),levels=[dec.min(), 0, dec.max()],
             cmap=mglearn.cm2, alpha=0.5)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")