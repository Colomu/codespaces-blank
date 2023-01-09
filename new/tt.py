import os
import pandas as pd
import numpy as np 

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from typing import Type
from typing import Dict, Any, Tuple, Type
from xgboost import XGBRegressor
Model = Type[Any] 

from math import sqrt
from itertools import product


# Import the load_airbnb function from the tabular_data module
from tabular_data import load_airbnb

# Load the Airbnb data with the Price column as the labels
features, labels = load_airbnb(label='Price_Night')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.80, test_size = 0.2, random_state=15)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_val.shape)
print(y_val.shape)


# Create a pipeline with a SimpleImputer for filling missing values
# and a StandardScaler for scaling the data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Create a ColumnTransformer to apply the numeric transformation
# to the numerical columns of the data
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, features.columns)
])

# Create a pipeline with the preprocessor and the SGDRegressor model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SGDRegressor(random_state=42))
])

# Fit the model to the training data
model.fit(X_train, y_train)

# Use the model to make predictions on the test data
predictions = model.predict(X_test)

# Calculate the mean squared error of the model
mse = mean_squared_error(y_test, predictions)
rmse = sqrt(mse)
print("RMSE:", rmse)

# Calculate the R^2 score for the training and test sets
r2_train = r2_score(y_train, model.predict(X_train))
r2_train = r2_score(y_train, model.predict(X_train))

print("R^2 Score:", r2_train)


def tune_regression_model_hyperparameters(X_train, y_train):
    kf = KFold(n_splits = 5)
    best_score = -np.inf
    best_params = {}
    
    for n_estimators in [100, 200, 500]:
        for max_depth in [3, 6, 9]:
            for gamma in [0.01, 0.1]:
                for learning_rate in [0.001, 0.01, 0.1, 1]:
                    params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "gamma": gamma,
                        "learning_rate": learning_rate,
                        "random_state": 2021
                    }
                    scores = []
                    for train_index, test_index in kf.split(X_train):
                        X_fold_train, X_fold_test = X_train[train_index], X_train[test_index]
                        y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]
                        model = XGBRegressor(**params)
                        model.fit(X_fold_train, y_fold_train)
                        y_pred = model.predict(X_fold_test)
                        scores.append(mean_squared_error(y_fold_test, y_pred))
                    mean_score = np.mean(scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params
    
    print("Best params:", best_params)
    print("Best MSE:", best_score)

# call the function
tune_regression_model_hyperparameters(X_train, y_train)
