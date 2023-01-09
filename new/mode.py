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

# Load the data with the "Price_Night" column as the label
features, labels = load_airbnb("Price_Night")

# Drop rows with missing values
features = features.dropna()
labels = labels.dropna()


# Create an instance of the SGDRegressor class
model = SGDRegressor()

# Train the model on the data
model.fit(features, labels)

# Load the new data from the CSV file
new_data = pd.read_csv("listing_clean.csv")

# Drop rows with missing values
new_data = new_data.dropna()


# Convert the data to a NumPy array
new_data = new_data.to_numpy()
new_data = new_data.values

# Make predictions on the new data using the trained model
predictions = model.predict(new_data)

# Print the predictions
print(predictions)



