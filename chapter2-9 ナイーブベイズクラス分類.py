# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:47:44 2020

@author: tikut
"""

import numpy as np
X = np.array([[0,1,0,1],
              [1,0,1,1],
              [0,0,0,1],
              [1,0,1,0]])
y = np.array([0,1,0,1])

counts = {}
for label in np.unique(y):
    #クラスに対してループ
    #それぞれの特徴量ごとに非ゼロの数を（加算で）数える
    counts[label] = X[y == label].sum(axis=0)
print("Feature counts:\n{}".format(counts))
