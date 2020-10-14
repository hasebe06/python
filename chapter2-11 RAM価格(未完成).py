# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:43:18 2020

@author: tikut
"""
import mglearn
import pandas as pd
import os
import matplotlib.pyplot as plt

ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH,
                                      "ram_price.csv"))

plt.semilogy(ram_prices.data, ram_prices.price)
plt.xlabel("Year")
plt.ylabel("Price in $/Mbyte")
