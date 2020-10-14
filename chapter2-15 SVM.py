# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:48:03 2020

@author: tikut
"""
import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
X,y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X,y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:,0], X[:,1], y)
#サポートベクタをプロットする
sv = svm.support_vectors_
#サポートベクタのクラスラベルはdual_coef_の正負によって与えられる。
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:,0], sv[:,1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")