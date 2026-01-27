"""utility_functions.py"""
# OPTIONAL
# In this file you can include help functions and/or classes that you want to use in your notebook
# It makes the code more organized and re-usable to have them on a separate .py file
# Each function and/or class should be separately documented with comments following this pattern:
# - Use: what does it do/serve
# - Inputs: input shape and description
# - Outputs: output shape and description


import os
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
