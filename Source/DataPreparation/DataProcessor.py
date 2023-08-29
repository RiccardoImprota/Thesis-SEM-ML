

# Importing libraries for data manipulation
import pandas as pd
import numpy as np


# Importing libraries for machine learning
import xgboost as xgb
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from joblib import dump,load

# Importing libraries for hyperparameter optimization
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# Display setting for exploration
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Used to run the R script
import subprocess
import os

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Check out if the environment is the correct Anaconda one
import sys
#print('environment: ',sys.executable)

# Set up directory to be the github repository
# requires git
os.getcwd()
output = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
path = output.decode('utf-8').strip()
#print('working directory: ',path)
os.chdir(path)



class DataProcessor:
    def __init__(self):
        """
        Initialize the DataProcessor object.
        """
        
        
    def read_df(self,csv_file='data\\processed\\df_reversed.csv'):
        """
        Reads a CSV file - for our models, it is assumed that the df is numeric
        and that the variables have already been reversed when necessary.

        Returns:
        pd.DataFrame: DataFrame of the read data.
        """
        self.csv_file = csv_file
        df = pd.read_csv(self.csv_file)
        
        # Convert all columns to nullable integer (Notice the "I" in "Int64)
        df = df.astype('Int64')
        
        self.reversed_df= df
        print('The dataframe was loaded')
        return df
        
    
    def split_data(self, stratify=None, test_size=0.3, random_state=None):
        """
        Splits the DataFrame into a training set and a test set.

        Parameters:
        stratify (str): The column(s) to use for stratification.

        Returns:
        pd.DataFrame, pd.DataFrame: Training data, Test data.
        """
        self.test_size = test_size
        self.random_state = random_state
    
        if stratify is not None:

            # Perform the split
            train_df, test_df = model_selection.train_test_split(
                self.reversed_df, 
                test_size=self.test_size, 
                random_state=self.random_state, 
                stratify=self.reversed_df[stratify]
            )
            
        else:
            train_df, test_df = model_selection.train_test_split(
                    self.reversed_df, 
                    test_size=self.test_size, 
                    random_state=self.random_state
                )
            
        self.train_df=train_df
        self.test_df=test_df
        print(f'A Train-Test split was performed with a test size of {self.test_size}')

        #display(train['eng_enthusiastic'].value_counts(normalize=True))
        #display(test['eng_enthusiastic'].value_counts(normalize=True))
        return train_df, test_df
    
    def save_data(self):
        """
        Saves the training set and the test set as CSV files.
        """
        self.train_df.to_csv('data\\processed\\factordatasets\\traindf.csv', index=False)
        self.test_df.to_csv('data\\processed\\factordatasets\\testdf.csv', index=False)

        print('Datasets were saved')
    
    def process_CFA(self, script_path='Source\\Defining Dimensions\\EGACFA.R'):
        """
        Given a R script path, it executes the R script.
        """
        
        # Rscript path
        r_script_path = "C:\\Program Files\\R\\R-4.3.1\\bin\\Rscript.exe"
        
        if os.path.isfile(script_path) != True:
            raise ValueError(f"Invalid script path.")


        print('Starting the CFA')
        # Run the command with arguments and capture output
        process_result = subprocess.run([r_script_path, script_path], capture_output=True, text=True)

        # Print the output
        #print("STDOUT:")
        #print(process_result.stdout)
        # Print the error if there's any
        if process_result.stderr:
            print("Error in the R code:")
            print(process_result.stderr)

    def read_cfadatasets(self):
        """
        Reads the CSV files that are generated by EGACFA.R

        Returns:
        pd.DataFrame, pd.DataFrame: CFA Training data, CFA Test data.
        """
        
        self.cfa_traindf = pd.read_csv('data\\processed\\factordatasets\\cfatrain.csv')
        self.cfa_testdf = pd.read_csv('data\\processed\\factordatasets\\cfatest.csv')
        # Convert all columns to nullable integer (Notice the "I" in "Int64)

        excluded_cols = ['ProfessionalSupport', 'JobOverload', 'Environmentalrisks', 'WorkAgency', 'WHO5', 'WorkEngagement']

        for col in self.cfa_traindf.columns:
            if col not in excluded_cols:
                self.cfa_traindf[col] = self.cfa_traindf[col].astype('Int64')

        for col in self.cfa_testdf.columns:
            if col not in excluded_cols:
                self.cfa_testdf[col] = self.cfa_testdf[col].astype('Int64')

        return self.cfa_testdf,self.cfa_testdf
    

    def process_datasets_pipeline(self,split_test_size=0.3,SeedSplit=None):
        """
        Using the functions of this class, executes a code that reads the initial dataframe, splits it into train/test, processes the factor scores and saves the new data into the
        the cfatrain.csv and cfatest.csv that are used in the read_cfadatasets() and dataprocess_WE_factor_scores() functions.
        """
        self.read_df()
        self.split_data(test_size=split_test_size,random_state=SeedSplit)
        self.save_data()
        self.process_CFA()

    def train_test_data_for_WEtarget(self, target_variable,Categories=False,combineseldom=False):
        """
        Prepares datasets where the target can be one of the following WorkEngagement dimensions: 
        'eng_timeflies', 'eng_enthusiastic', 'eng_energy', 'WorkEngagement' or its factor scores.

        Args:
        target_variable (str): The target variable for the prediction. Must be one of 
        'eng_timeflies', 'eng_enthusiastic', 'eng_energy', 'WorkEngagement'.


        Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame: X_train, y_train, X_test , y_test

        Raises:
        ValueError: If target_variable is not one of the valid options.
        """
        valid_targets = {'eng_timeflies', 'eng_enthusiastic', 'eng_energy', 'WorkEngagement'}
        if target_variable not in valid_targets:
            raise ValueError(f"Invalid target_variable. Expected one of: {valid_targets}")

        self.read_cfadatasets()

        temp_traindf=self.cfa_traindf
        temp_testdf=self.cfa_testdf

        # This code is useful if some models require variables to be of type "category" to function properly
        if Categories==True:
            # Define a list of columns to exclude
            exclude = ['ProfessionalSupport', 'JobOverload', 'Environmentalrisks', 
                       'WorkAgency', 'WHO5', 'seniority', 'usual_hours_week']
            
            # Select all columns except those in the exclude list and convert them to 'category'
            temp_traindf[temp_traindf.columns.difference(exclude)] = temp_traindf[temp_traindf.columns.difference(exclude)].astype('category')
            temp_testdf[temp_testdf.columns.difference(exclude)] = temp_testdf[temp_testdf.columns.difference(exclude)].astype('category')

        X_train = temp_traindf.drop(['ID', 'SurveyCombination_M1', 'SurveyCombination_M2', 'Country', 'gender_recoded','age','eng_timeflies','eng_enthusiastic','eng_energy','WorkEngagement'], axis=1)
        y_train = temp_traindf[target_variable]
        X_test = temp_testdf.drop(['ID', 'SurveyCombination_M1', 'SurveyCombination_M2', 'Country', 'gender_recoded','age','eng_timeflies','eng_enthusiastic','eng_energy','WorkEngagement'], axis=1)
        y_test = temp_testdf[target_variable]

        # Given that the distributions of "never" and "rarely" can be quite low, this argument merges them in a single column
        if combineseldom==True:
            columns_to_update = {'eng_timeflies', 'eng_enthusiastic', 'eng_energy'}

            if target_variable in columns_to_update:
                print('tesr')
                y_train = y_train.replace(1, 2) - 1
                y_test = y_test.replace(1, 2) - 1


        return X_train.astype("float64"),y_train.astype("float64"),X_test.astype("float64"),y_test.astype("float64")