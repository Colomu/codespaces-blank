import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.stats as stats


def remove_rows_with_missing_ratings(df):
    """Remove rows with missing values in the rating columns."""
    return df[~df[['Cleanliness_rating', 'Accuracy_rating', 'Location_rating', 'Check-in_rating', 'Value_rating']].isnull().any(axis=1)]

def combine_description_strings(df):
    """Combine the list items in the "Description" column into the same string."""
    df['Description'] = df['Description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    return df[df['Description'].notna()]

def set_default_feature_values(df):
    """Replace empty values in the "guests", "beds", "bathrooms", and "bedrooms" columns with the number 1."""
    df[['guests', 'beds', 'bathrooms', 'bedrooms']] = df[['guests', 'beds', 'bathrooms', 'bedrooms']].astype(float).fillna(1)
    return df


def remove_outliers(df, numerical_columns):
    """Remove rows with extreme values in the numerical columns."""
    df = df[(np.abs(stats.zscore(df[numerical_columns].values)) < 3).all(axis=1)]
    return df


def standardize_data(df, numerical_columns):
    """Standardize the numerical columns."""
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def impute_missing_values(df, numerical_columns):
    """Impute missing values in the numerical columns."""
    imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
    return df

def clean_tabular_data(df, numerical_columns):
    """Clean the data in a tabular dataset."""
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    df = remove_outliers(df, numerical_columns)
    df = standardize_data(df, numerical_columns)
    df = impute_missing_values(df, numerical_columns)
    return df

def load_airbnb(label: str, features: list) -> tuple:
    """Load the Airbnb data and return features and labels as a tuple.

    Parameters
    ----------
    label : str
        The name of the column to use as labels.
    features : list
        The names of the columns to use as features.

    Returns
    -------
    tuple
        A tuple of the form (features, labels).
    """
    # Load the raw data in using pandas
    df = pd.read_csv('listing_clean.csv', index_col='id', skipinitialspace=True)

    # Select the specified columns
    features = df[features]
    # Select the label column as the labels
    labels = df[label]

    return features, labels



if __name__ == "__main__":
    # Load the raw data in using pandas
    df = pd.read_csv('listing.csv')
    # Call clean_tabular_data on it
    numerical_columns = ['guests', 'beds', 'bathrooms', 'bedrooms', 'Price_Night']
    df = clean_tabular_data(df, numerical_columns)
    # Save the processed data as clean_tabular_data.csv in the same folder as you found the raw tabular data.
    df.to_csv('listing_clean.csv', index=False)
    # Load the Airbnb data with the Price column as the labels
    features, labels = load_airbnb(label='Price_Night', features=['guests', 'beds', 'bathrooms', 'bedrooms'])
    print(features.head())
    print(labels.head())



