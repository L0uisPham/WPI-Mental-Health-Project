""" Models used in analysis """


import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, make_scorer

# Supress Warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
pd.options.mode.chained_assignment = None 


MODELS = {
    'LogR': {
        'clf': LogisticRegression,
        'hyperparameters': {
            'penalty': ['l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'max_iter': [100, 1000, 5000]
        }
    },
    'SVM': {
        'clf': SVC,
        'hyperparameters': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1, 10, 100]
        }
    },
    'RFC': {
        'clf': RandomForestClassifier,
        'hyperparameters': {
            'n_estimators': [250, 500, 1000, 1250],
            'max_depth': [1, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'GradientBoosting': {
        'clf': GradientBoostingClassifier,
        'hyperparameters': {
            'n_estimators': [10, 50, 100, 250],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'learning_rate': [0.001, 0.01, 0.1, 1]
        }
    },
    'DecisionTree': {
        'clf': DecisionTreeClassifier,
        'hyperparameters': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'class_weight': [None, 'balanced']
        }
    }
}