## Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

## functions
# define a function to preprocess origin column
def preprocess_origin_cols(df):
    df_copy = df.copy()
    df_copy["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
    return df_copy


# custom attribute adder
from sklearn.base import BaseEstimator, TransformerMixin

acc_ix, hpower_ix, cyl_ix = 4, 2, 0

class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True): # no *args or **kargs
        self.acc_on_power = acc_on_power
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        acc_on_cyl = X[:, acc_ix] / X[:, cyl_ix]
        if self.acc_on_power:
            acc_on_power = X[:, acc_ix] / X[:, hpower_ix]
            return np.c_[X, acc_on_power, acc_on_cyl]
        
        return np.c_[X, acc_on_cyl]

# pipeline to transform data

def pipeline_transformer(data):
    '''
    Complete transformation pipeline for both nuerical and categorical data.
    Argument:
        data: original dataframe 
    Returns:
        prepared_data: transformed data, ready to use
    '''
    
    # preprocess the origin column
    preprocessed_data = preprocess_origin_cols(data)

    # categorical attributes and transformer pipeline
    cat_feature = ["Origin"]
    cat_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    # numeric features and transformer pipeline
    num_features = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year']
    num_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy="median")),
      ('attrs_adder', CustomAttrAdder()),
      ('std_scaler', StandardScaler()),
    ])

    # Setup preprocessing steps (fill missing values, then convert to numbers)
    preprocessor = ColumnTransformer(transformers=[
        ("cat", cat_transformer, cat_feature),
        ("num", num_transformer, num_features),
        ])
    
    # apply the transformations on data
    transformed_data = preprocessor.fit_transform(preprocessed_data)
    return transformed_data
    

# define a function to take input data and predict the mpg value using our best model
def predict_mpg(config, model):
    
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    
    transformed_df = pipeline_transformer(df)
    y_pred = model.predict(transformed_df)
    return y_pred