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
from typing import Type
from typing import Dict, Any, Tuple, Type
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


#X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.80, test_size = 0.2, random_state=15)

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
    ('num', numeric_transformer, features.column)
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


def custom_tune_regression_model_hyperparameters(
    model_class: Type[Any], 
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    hyperparameters: Dict[str, Any]
) -> Tuple[Type[Any], Dict[str, Any], Dict[str, Any]]:

    # Load the data
    X_train = (569,10)
    y_train = (569)
    X_val = (143,10)
    y_val = (143)
    X_test = (178)
    y_test = (178)



    # Initialise variables to keep track of the best model and its performance
    best_model = None
    best_hyperparameters = {}
    best_performance = {'validation_RMSE': float('inf')}

    # Iterate over all combinations of hyperparameter values
    for hyperparameter_values in product(*hyperparameters.values()):
        # Create a new model with the current combination of hyperparameter values
        model = model_class(**dict(zip(hyperparameters.keys(), hyperparameter_values)))

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        val_predictions = model.predict(X_val)

        # Calculate the RMSE on the validation set
        rmse = sqrt(mean_squared_error(y_val, val_predictions))

        # Update the best model if the current model has a lower RMSE
        if rmse < best_performance['validation_RMSE']:
            best_model = model
            best_hyperparameters = dict(zip(hyperparameters.keys(), hyperparameter_values))
            best_performance = {'validation_RMSE': rmse}

    # Retrain the best model on the combined training and validation sets
    best_model.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))

    # Evaluate the best model on the test set
    test_predictions = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_evs = explained_variance_score(y_test, test_predictions)

    # Update the best performance dictionary with the test set performance metrics
    best_performance.update({
        'test_mse': test_mse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_evs': test_evs
    })

    return best_model, best_hyperparameters, best_performance

    # Load the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Define the model class and hyperparameters to be tuned
    model_class = RandomForestRegressor
    hyperparameters = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'random_state': [0]
    }

    # Tune the model
    best_model, best_hyperparameters, best_performance = custom_tune_regression_model_hyperparameters(
        model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparameters
    )

    # Print the best model and its performance
    print('Best model:', best_model)
    print('Best hyperparameters:', best_hyperparameters)
    print('Best performance:', best_performance)

    # Save the best model to a file
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # Save the best hyperparameters to a file
    with open('best_hyperparameters.pkl', 'wb') as f:
        pickle.dump(best_hyperparameters, f)

    # Save the best performance to a file
    with open('best_performance.pkl', 'wb') as f:
        pickle.dump(best_performance, f)


    best_model, best_hyperparameters, best_performance = custom_tune_regression_model_hyperparameters(model_class=SGDRegressor, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, hyperparameters=hyperparameters)

