# Gradient boosting for Work Engagement Latent variable regression

# Importing libraries for data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing libraries for machine learning
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump,load
import shap


# Importing libraries for hyperparameter optimization
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# Display setting for exploration
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

class GBoostRegression:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = None
        self.bestmodel = None
        self.results = None
        self.shap_values = None
        self.explainer_xgb  = None

    def train(self):
        # Define the hyperparameter search space
        search_spaces = {
        'learning_rate': Real(0.01, 0.3, 'log-uniform'),  # Typical range for learning rates; 'log-uniform' because smaller changes matter more for learning rates
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
            xgb.XGBRegressor(),
            search_spaces,
            n_iter=60,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
            verbose=1
        )

        print('Starting the XGBRegressor training')
        opt.fit(self.x_train, self.y_train)
        
        self.model = opt.best_estimator_

        self.y_pred = self.model.predict(self.x_test)
        
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        
        self.results = {
            "best_params": opt.best_params_,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

        # SHAP
        self.explainer_xgb  = shap.Explainer(self.model)
        self.shap_values = self.explainer_xgb (self.x_test)

    def get_results(self):
        return self.results
        
    def get_shap_values(self):
        return self.shap_values
    
