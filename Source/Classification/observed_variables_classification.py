# Gradient boosting for Work Engagement Observed dimensions

# Importing libraries for data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing libraries for machine learning
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump,load
import shap
import json

# Importing libraries for hyperparameter optimization
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# Display setting for exploration
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

try:
    # Set up directory to be the github repository
    # requires git
    import os
    import subprocess
    os.getcwd()
    output = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
    path = output.decode('utf-8').strip()
except:
    path=None



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

class GBoostClassification:
    """
    XGB Classification with Bayesian optimization and SHAP value computation.
    """
    
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Initializes data, imputer, and prepares data.
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_pred = None
        self.y_pred = None
        self.model = None
        self.results = {}
        self.shap_values = None
        self.explainer_xgb  = None

    def train(self,scoringmetric='neg_mean_squared_error',verbosity=1,n_iter=60,cv=5,computeshap=True):
        """
        Train the XGB classifier with Bayesian optimization.
        Also, computes SHAP values for interpretability.
        """

        self.scoring_metric = scoringmetric
        self.n_iter = n_iter
        self.cv = cv

        # Define the hyperparameter search space
        search_spaces = {
        'learning_rate': Real(0.01, 0.1, 'log-uniform'),  # Typical range for learning rates; 'log-uniform' because smaller changes matter more for learning rates
        'n_estimators': Integer(100, 400),  # Allow for a wide range of trees, but don't go too low
        'max_depth': Integer(3, 10),  # Range allowing for deeper trees which might be needed for complex datasets
        'min_child_weight': Integer(1, 10),  # Default is 1, but higher values make the algorithm more conservative
        'subsample': Real(0.5, 1.0, 'uniform'),  # Typical range to prevent overfitting
        'colsample_bytree': Real(0.5, 1.0, 'uniform'),  # Subsample ratio of columns when constructing each tree
        #'colsample_bylevel': Real(0.5, 1.0, 'uniform'),  # Subsample ratio of columns for each level in the tree
        'gamma': Real(0.0, 0.5, 'uniform'),  # Minimum loss reduction; 0 means no regularization, but we don't allow it to be too high
        'reg_lambda': Real(0.01, 5.0, 'uniform'),  # L2 regularization; keep within reasonable range
        'reg_alpha': Real(0.0, 1.0, 'uniform'),  # L1 regularization; again, we want to keep this within a typical range
        }


        opt = BayesSearchCV(
            xgb.XGBClassifier(objective='multi:softprob'),
            search_spaces,
            n_iter=n_iter,
            cv=cv,
            scoring=scoringmetric,
            random_state=42,
            verbose=verbosity
        )

        print('Starting the XGBRegressor training')
        opt.fit(self.x_train, self.y_train,early_stopping_rounds=10)
        
        self.model = opt.best_estimator_

        self.y_pred = self.model.predict(self.x_test)
        self.y_train_pred = self.model.predict(self.x_train)

        self._compute_training_error()
        self._compute_metrics()
        if computeshap==True:
            self._compute_shap_values()