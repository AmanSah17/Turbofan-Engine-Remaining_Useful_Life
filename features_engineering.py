#!/usr/bin/env python
# coding: utf-8

# # Features Engineering
# In this notebook we experiment with some features engineering:
# 
# using TSFresh
# 
# and using ROCKET (Random Convolutional Kernel Transform)
# 
# NOTE: Before starting exploring this notebook, I recommend checking 1-EDA.ipynb notebook first - it contains Exploratory Data Analysis and will help you get some understanding of the datasets.

# 

# In[2]:


import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ## First, let's read aircraft engines datasets, we start with "FD001" dataset, which has:
# 
# 100 engines time series in TRAIN set
# 
# 100 engines time series in TEST set
# 
# 1 Fault condition
# 
# 1 Operating condition

# In[11]:


from utils2 import read_dataset, calculate_RUL, SENSOR_COLUMNS

train, test, test_rul = read_dataset("FD001")

train["rul"] = calculate_RUL(train, upper_threshold=125)

print(f"train.shape = {train.shape}")
train.head(2)


# In[6]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold


class LowVarianceFeaturesRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0):
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=threshold)

    def fit(self, X):
        self.selector.fit(X)
        return self

    def transform(self, X):
        X_t = self.selector.transform(X)
        droped_features = X.columns[~self.selector.get_support()]
        print(f"Droped low variance features: {droped_features.to_list()}")
        return pd.DataFrame(X_t, columns=self.selector.get_feature_names_out())


# In[7]:


from utils2 import SENSOR_COLUMNS


class ScalePerEngine(BaseEstimator, TransformerMixin):
    """
    Scale individual engines time series with respect to its start.
    Substract firts `n_first_cycles` AVG values from time series.
    """

    def __init__(self, n_first_cycles=20, sensors_columns=SENSOR_COLUMNS):
        self.n_first_cycles = n_first_cycles
        self.sensors_columns = sensors_columns

    def fit(self, X):
        return self

    def transform(self, X):
        self.sensors_columns = [x for x in X.columns if x in self.sensors_columns]

        init_sensors_avg = (
            X[X["time_cycles"] <= self.n_first_cycles]
            .groupby(by=["unit"])[self.sensors_columns]
            .mean()
            .reset_index()
        )

        X_t = X[X["time_cycles"] > self.n_first_cycles].merge(
            init_sensors_avg, on=["unit"], how="left", suffixes=("", "_init_v")
        )

        for SENSOR in self.sensors_columns:
            X_t[SENSOR] = X_t[SENSOR] - X_t["{}_init_v".format(SENSOR)]

        drop_columns = X_t.columns.str.endswith("init_v")
        return X_t[X_t.columns[~drop_columns]]


# In[ ]:





# In[ ]:




