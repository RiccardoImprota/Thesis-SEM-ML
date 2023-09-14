# Gradient boosting for Work Engagement Latent variable regression

# Importing libraries for data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing libraries for machine learning
import xgboost as xgb
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

class GBoostRegression:
    """
    XGB Regressor with Bayesian optimization and SHAP value computation.
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


    def train(self,scoringmetric='neg_mean_squared_error',verbosity=1,n_iter=60,cv=5,computeshap=True):
        """
        Train the XGB regressor with Bayesian optimization.
        Also, computes SHAP values for interpretability.
        """

        self.scoring_metric = scoringmetric
        self.n_iter = n_iter
        self.cv = cv


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
            xgb.XGBRegressor(),
            search_spaces,
            n_iter=n_iter,
            cv=cv,
            scoring=scoringmetric,
            random_state=42,
            verbose=verbosity
        )

        print('Starting the XGBRegressor training')
        opt.fit(self.x_train, self.y_train)
        
        self.model = opt.best_estimator_

        self.y_pred = self.model.predict(self.x_test)
        self.y_train_pred = self.model.predict(self.x_train)

        self._compute_training_error()
        self._compute_metrics()
        if computeshap==True:
            self._compute_shap_values()

    def _compute_training_error(self):
        """
        Compute the training error.
        """
        train_mse = mean_squared_error(self.y_train, self.y_train_pred)
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        self.results["train_mse"] = train_mse
        self.results["train_r2"] = train_r2


    def _compute_metrics(self):
        """
        Compute evaluation metrics.
        """
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        
        self.results["best_params"] = self.model.get_params()
        self.results["mse"] = mse
        self.results["mae"] = mae
        self.results["r2"] = r2

    def _compute_shap_values(self):
        """
        Compute SHAP values.
        """
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer(self.x_test)

    def get_results(self,verbose=0):
        """
        Return the evaluation metrics.
        """
        if verbose>0:
            print("Best Parameters:")
            print(self.results['best_params'])
    
        
            print("\n\nMetrics:")
            print(f"MSE: {self.results['mse']}")
            print(f"MAE: {self.results['mae']}")
            print(f"R^2: {self.results['r2']}")
        return self.results
        
    def get_shap_values(self):
        """
        Return the SHAP values.
        """
        return self.shap_values
    
    def save_model(self, directory='Models\\Regression', model_name=None):
        """
        Save the trained model and associated metadata to the specified directory.
        
        Args:
            directory (str): Path to the directory where the model and its metadata will be saved.
            model_name (str): Base name to use for saving the model and metadata files.
        """
        # Ensure directory exists
        if not os.path.exists(directory):
            raise ValueError(f"The specified directory '{directory}' does not exist.")
        
        if model_name==None:
            raise ValueError(f"A model name was not specified.")

        # Save model to .pkl file
        model_path = os.path.join(directory, f"{model_name}.pkl")
        dump(self.model, model_path)

        # Save model metadata to .json file
        metadata = {
            "model": "XGB Regressor",
            "results": self.results,
            "training_details": {
                "scoring_metric": self.scoring_metric,
                "n_iter": self.n_iter,
                "cv": self.cv
            }
        }
        metadata_path = os.path.join(directory, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

class RFRegression:
    """
    Random Forest Regressor with Bayesian optimization and SHAP value computation.
    """
    
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Initializes data, imputer, and prepares data.
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = None
        self.y_train_pred = None
        self.model = None
        self.results = {}
        self.shap_values = None
        self.explainer_xgb  = None
        self.median_imputation()

    def median_imputation(self):
        """
        Applies median imputation to the training and test data.
        """
        imputer = SimpleImputer(strategy='median')
        self.feature_names = self.x_train.columns # Save the column names before imputation
        self.x_train = pd.DataFrame(imputer.fit_transform(self.x_train), columns=self.feature_names)
        self.x_test = pd.DataFrame(imputer.transform(self.x_test), columns=self.feature_names)


    def train(self,verbosity=1,n_iter=60,cv=5,computeshap=True):
        """
        Train the Random Forest regressor with Bayesian optimization.
        Also, computes SHAP values for interpretability.
        """
        self.n_iter = n_iter
        self.cv = cv


        # Define the hyperparameter search space
        search_spaces = {
            'n_estimators': Integer(10, 100),
            'max_depth': Integer(1, 10),
            'min_samples_split': Integer(2, 15),
            'min_samples_leaf': Integer(1, 15)
        }

        opt = BayesSearchCV(
            RandomForestRegressor(),
            search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=42,
            verbose=verbosity
        )

        print('Starting the Random Forest training with median imputation')
        opt.fit(self.x_train, self.y_train)
        
        self.model = opt.best_estimator_

        self.y_pred = self.model.predict(self.x_test)
        self.y_train_pred = self.model.predict(self.x_train)

        self._compute_training_error()
        self._compute_metrics()
        if computeshap==True:
            self._compute_shap_values()

    def _compute_training_error(self):
        """
        Compute the training error.
        """
        train_mse = mean_squared_error(self.y_train, self.y_train_pred)
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        self.results["train_mse"] = train_mse
        self.results["train_r2"] = train_r2

    def _compute_metrics(self):
        """
        Compute evaluation metrics.
        """
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        

        self.results["best_params"] = self.model.get_params()
        self.results["mse"] = mse
        self.results["mae"] = mae
        self.results["r2"] = r2

    def _compute_shap_values(self):
        """
        Compute SHAP values.
        """
        explainer = shap.Explainer(self.model)
        self.shap_values = explainer(pd.DataFrame(self.x_test, columns=self.feature_names))

    def get_results(self,verbose=0):
        """
        Return the evaluation metrics.
        """

        if verbose>0:
            print("Best Parameters:")
            print(self.results['best_params'])
    
        
            print("\n\nMetrics:")
            print(f"MSE: {self.results['mse']}")
            print(f"MAE: {self.results['mae']}")
            print(f"R^2: {self.results['r2']}")
        return self.results
        
    def get_shap_values(self):
        """
        Return the SHAP values.
        """
        return self.shap_values
    
    
    def save_model(self, directory='Models\\Regression', model_name=None):
        """
        Save the trained model and associated metadata to the specified directory.
        
        Args:
            directory (str): Path to the directory where the model and its metadata will be saved.
            model_name (str): Base name to use for saving the model and metadata files.
        """
        # Ensure directory exists
        if not os.path.exists(directory):
            raise ValueError(f"The specified directory '{directory}' does not exist.")
        
        if model_name==None:
            raise ValueError(f"A model name was not specified.")

        # Save model to .pkl file
        model_path = os.path.join(directory, f"{model_name}.pkl")
        dump(self.model, model_path)

        # Save model metadata to .json file
        metadata = {
            "model": "Random Forest Regressor",
            "results": self.results,
            "training_details": {
                "n_iter": self.n_iter,
                "cv": self.cv
            }
        }
        metadata_path = os.path.join(directory, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
    

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

class LinearRegressionModel:
    """
    Linear Regression model with SHAP values computation and evaluation metrics.
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
        self.results = {}
        self.model = None
        self.shap_values = None


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

        self.model = LinearRegression()
        print('Starting the Linear Regression training')
        self.model.fit(self.x_train, self.y_train)
        
        self.y_pred = self.model.predict(self.x_test)
        self.y_train_pred = self.model.predict(self.x_train)

        self._compute_training_error()
        self._compute_metrics()
        if computeshap==True:
            self._compute_shap_values()

        if computeshap==True:
            self._compute_shap_values()

    def _compute_training_error(self):
        """
        Compute the training error.
        """
        train_mse = mean_squared_error(self.y_train, self.y_train_pred)
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        self.results["train_mse"] = train_mse
        self.results["train_r2"] = train_r2


    def _compute_metrics(self):
        """
        Compute evaluation metrics.
        """
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        
        self.results["mse"] = mse
        self.results["mae"] = mae
        self.results["r2"] = r2

    def _compute_shap_values(self):
        """
        Compute SHAP values.
        """
        explainer = shap.Explainer(self.model, self.x_train)
        self.shap_values = explainer(self.x_test)


    def get_results(self,verbose=0):
        """
        Return the evaluation metrics.
        """

        if verbose>0:
            print("Best Parameters:")
            print(self.results['best_params'])
    
        
            print("\n\nMetrics:")
            print(f"MSE: {self.results['mse']}")
            print(f"MAE: {self.results['mae']}")
            print(f"R^2: {self.results['r2']}")
        return self.results
    
    def get_shap_values(self):
        """
        Return the SHAP values.
        """
        return self.shap_values
    


    def save_model(self, directory='Models\\Regression', model_name=None):
        """
        Save the trained model and associated metadata to the specified directory.
        
        Args:
            directory (str): Path to the directory where the model and its metadata will be saved.
            model_name (str): Base name to use for saving the model and metadata files.
        """
        # Ensure directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        if model_name==None:
            raise ValueError(f"A model name was not specified.")

        # Save model to .pkl file
        model_path = os.path.join(directory, f"{model_name}.pkl")
        dump(self.model, model_path)

        # Save model metadata to .json file
        metadata = { "model": "Linear Regression",
            "results": self.results
        }
        metadata_path = os.path.join(directory, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


class ElasticLinear:
    """
    Linear Regression model with SHAP values computation and evaluation metrics.
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
        self.results = {}
        self.model = None
        self.shap_values = None


    def median_imputation(self, standardscaling=True):
        """
        Applies median imputation to the training and test data.
        """
        imputer = SimpleImputer(strategy='median')
        self.feature_names = self.x_train.columns # Save the column names before imputation
        self.x_train = pd.DataFrame(imputer.fit_transform(self.x_train), columns=self.feature_names)
        self.x_test = pd.DataFrame(imputer.transform(self.x_test), columns=self.feature_names)

        if standardscaling:
            scaler = StandardScaler()
            self.x_train = pd.DataFrame(scaler.fit_transform(self.x_train), columns=self.x_train.columns)
            self.x_test = pd.DataFrame(scaler.transform(self.x_test), columns=self.x_test.columns)


    def train(self,verbosity=1,n_iter=40,cv=5,computeshap=True):
        """
        Train the Linear Regression model and compute SHAP values.
        """

        search_spaces = {
        'alpha': (0.0001, 1.0, 'log-uniform'),
        'l1_ratio': (0.01, 1.0)
        }

        opt = BayesSearchCV(
            ElasticNet(),
            search_spaces,
            n_iter=n_iter,
            cv=cv,
            random_state=42,
            verbose=verbosity
        )


        opt.fit(self.x_train, self.y_train)
        self.model = opt.best_estimator_

        self.y_pred = self.model.predict(self.x_test)
        self.y_train_pred = self.model.predict(self.x_train)

        self._compute_training_error()
        self._compute_metrics()
        if computeshap==True:
            self._compute_shap_values()

    def _compute_training_error(self):
        """
        Compute the training error.
        """
        train_mse = mean_squared_error(self.y_train, self.y_train_pred)
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        self.results["train_mse"] = train_mse
        self.results["train_r2"] = train_r2


    def _compute_metrics(self):
        """
        Compute evaluation metrics.
        """
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        
        self.results["mse"] = mse
        self.results["mae"] = mae
        self.results["r2"] = r2

    def _compute_shap_values(self):
        """
        Compute SHAP values.
        """
        explainer = shap.Explainer(self.model, self.x_train)
        self.shap_values = explainer(self.x_test)


    def get_results(self,verbose=0):
        """
        Return the evaluation metrics.
        """

        if verbose>0:
            print("Best Parameters:")
            print(self.model)
    
        
            print("\n\nMetrics:")
            print(f"MSE: {self.results['mse']}")
            print(f"MAE: {self.results['mae']}")
            print(f"R^2: {self.results['r2']}")
        return self.results
    
    def get_shap_values(self):
        """
        Return the SHAP values.
        """
        return self.shap_values
