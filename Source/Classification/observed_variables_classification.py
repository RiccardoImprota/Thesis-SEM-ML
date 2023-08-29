# Gradient boosting for Work Engagement Observed dimensions

# Importing libraries for data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing libraries for machine learning
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer,KNNImputer
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
        'colsample_bylevel': Real(0.5, 1.0, 'uniform'),  # Subsample ratio of columns for each level in the tree
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

        print('Starting the XGBClassifier training')

        self.y_train = self.y_train-1
        opt.fit(self.x_train, self.y_train)
        self.model = opt.best_estimator_
        self.y_train = self.y_train + 1

        self.y_pred = self.model.predict(self.x_test)
        self.y_pred = self.y_pred + 1
        self.y_proba = self.model.predict_proba(x_test)

        self.y_train_pred = self.model.predict(self.x_train)

        self._compute_training_error()
        self._compute_metrics()

    def _compute_training_error(self):
        """
        Compute the training error.
        """
        train_accuracy = accuracy(self.y_train, self.y_train_pred)
        train_precision = precision(self.y_train, self.y_train_pred, average='weighted',zero_division=np.nan)
        train_recall = recall_score(self.y_train, self.y_train_pred, average='weighted',zero_division=np.nan)
        self.results["train_accuracy"] = train_accuracy
        self.results["train_precision"] = train_precision
        self.results["train_recall"] = train_recall


    def _compute_metrics(self):
        """
        Compute evaluation metrics.
        """
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average='weighted',zero_division=np.nan)
        recall = recall_score(self.y_test, self.y_pred, average='weighted',zero_division=np.nan)
        
        self.results["best_params"] = self.model.get_params()
        self.results["accuracy"] = accuracy
        self.results["precision"] = precision
        self.results["recall"] = recall

        #print(classification_report(y_test, y_pred))

    def get_results(self,verbose=0):
        """
        Return the evaluation metrics.
        """
        return self.results

    def get_probas(self):
        """
        Return the probabilities.
        """
        return self.y_score




####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

class LogisticRegressionClass:
    """
    Logistic Regression model and evaluation metrics.
    """
    
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Initializes data and prepares for training.
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_pred = None
        self.y_pred = None
        self.model = None
        self.results = {}


    def median_imputation(self):
        """
        Applies median imputation to the training and test data.
        """
        imputer = SimpleImputer(strategy='median')
        self.feature_names = self.x_train.columns # Save the column names before imputation
        self.x_train = pd.DataFrame(imputer.fit_transform(self.x_train), columns=self.feature_names)
        self.x_test = pd.DataFrame(imputer.transform(self.x_test), columns=self.feature_names)



    def train(self,computeshap=True):
        """
        Train the Linear Regression model and compute SHAP values.
        """
        self.model = LogisticRegression()
        print('Starting the Linear Regression training')
        self.model.fit(self.x_train, self.y_train)
        
        self.y_pred = self.model.predict(self.x_test)
        self.y_train_pred = self.model.predict(self.x_train)

        self._compute_training_error()
        self._compute_metrics()
        if computeshap==True:
            self._compute_shap_values()


        print('Starting the Logistic Regression')
        self.model = LogisticRegression()
        self.y_train = self.y_train-1
        self.model.fit(self.x_train, self.y_train)
        self.y_train = self.y_train + 1


        self.y_pred = self.model.predict(self.x_test)
        self.y_pred = self.y_pred + 1
        self.y_proba = self.model.predict_proba(x_test)

        self.y_train_pred = self.model.predict(self.x_train)

        self._compute_training_error()
        self._compute_metrics()

    def _compute_training_error(self):
        """
        Compute the training error.
        """
        train_accuracy = accuracy(self.y_train, self.y_train_pred)
        train_precision = precision(self.y_train, self.y_train_pred, average='weighted',zero_division=np.nan)
        train_recall = recall_score(self.y_train, self.y_train_pred, average='weighted',zero_division=np.nan)
        self.results["train_accuracy"] = train_accuracy
        self.results["train_precision"] = train_precision
        self.results["train_recall"] = train_recall


    def _compute_metrics(self):
        """
        Compute evaluation metrics.
        """
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average='weighted',zero_division=np.nan)
        recall = recall_score(self.y_test, self.y_pred, average='weighted',zero_division=np.nan)
        
        self.results["best_params"] = self.model.get_params()
        self.results["accuracy"] = accuracy
        self.results["precision"] = precision
        self.results["recall"] = recall

        #print(classification_report(y_test, y_pred))

    def get_results(self,verbose=0):
        """
        Return the evaluation metrics.
        """
        return self.results

    def get_probas(self):
        """
        Return the probabilities.
        """
        return self.y_proba


