#!/usr/bin/env python

# import libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

raw_train = pd.read_csv('train.csv')
raw_test = pd.read_csv('test.csv')
raw_train.head()

raw_train.shape

raw_train.nunique()

train_df = raw_train.copy()
test_df = raw_test.copy()

train_df.info()

test_df.info() # only for prediction

train_y = train_df['Loan_Status'].copy()

train_df.drop(columns = ['Loan_Status'],inplace=True)

train_df.info()

# Dropping the unncessary columns
train_df.drop(columns='Loan_ID',inplace=True)
test_df.drop(columns='Loan_ID',inplace=True)

train_df.columns

# Duplicates --> no duplicates
train_df[train_df.duplicated()]

test_df[test_df.duplicated()]

test_df.drop_duplicates(inplace=True)

test_df[test_df.duplicated()]

# Missing Value analysis
train_df.isna().sum()

train_df.info()

train_df.columns

train_df.nunique()

# Numeric --> mean
# Categorical --> mode
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term']

cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
           'Credit_History', 'Property_Area']

cat_imputer = SimpleImputer(strategy="most_frequent")
cat_imputer.fit(train_df[cat_cols])

train_df[cat_cols] = cat_imputer.transform(train_df[cat_cols])
test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols]) 

num_imputer = SimpleImputer(strategy="mean")
num_imputer.fit(train_df[num_cols])

train_df[num_cols] = num_imputer.transform(train_df[num_cols])
test_df[num_cols] = num_imputer.transform(test_df[num_cols]) 

# Missing Value analysis
train_df.isna().sum()

train_df.head()

# preprocessing as per the domain knowledge
train_df['ApplicantIncome'] = train_df['ApplicantIncome'] + train_df['CoapplicantIncome']
test_df['ApplicantIncome'] = test_df['ApplicantIncome'] + test_df['CoapplicantIncome']

# drop the co-applicant income columne
train_df.drop(columns='CoapplicantIncome',inplace=True)
test_df.drop(columns='CoapplicantIncome',inplace=True)

train_df.head()

# Application of Label Encoder
train_df.nunique()

train_df.Dependents.unique() # Ordinal data --> label Encoder

train_df.Property_Area.unique()

for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

train_df.head()

num_cols.remove('CoapplicantIncome')

num_cols

cat_cols

num_cols

# log transformation
train_df[num_cols] = np.log(train_df[num_cols])
test_df[num_cols] = np.log(test_df[num_cols])

train_df.columns

#scaling
minmax = MinMaxScaler()
train_df = minmax.fit_transform(train_df)
test_df = minmax.transform(test_df)

# Building the Model
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train_df,train_y,test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)

y_pred_test = log.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred_test)
print(f"Accuracy is {acc}")


# # Serialization and Deserialization 
# serialization & deserialization
import joblib
joblib.dump(log,"my_trained_model_v1.pkl")

#deserialization
final_model = joblib.load("my_trained_model_v1.pkl")

final_model.intercept_, final_model.coef_

log.intercept_, log.coef_


# ## Package and Modules
import PackageA

from PackageA import f1

f1.print_something()

from PackageA import f2

f2.print_something()

from PackageA.f1 import print_something as f1p

f1p()

from PackageA.SubPackageA import f3

f3.print_something()

from PackageA.SubPackageB import f5
f5.print_something()

# # Adding the System Path
import sys

sys.path

sys.path.append('/Users/nachiketh/Desktop/') # mac or linux

#sys.path.append('C:\\path\\to\\dir') # windows

# #  Getting the Parent Directory
import PackageA

PackageA.__file__

import pathlib
pathlib.Path(PackageA.__file__).resolve().parent

# # Create Custom Data Transformers
# Key thing --> Inherit - BaseEstimator, TransformerMixin
# implement fit and transform
# accept input with __init__ method

from sklearn.base import BaseEstimator,TransformerMixin

class DemoTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        return X

# Numerical Imputation - mean
from sklearn.base import BaseEstimator,TransformerMixin

class MeanImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        self.mean_dict = {}
        for col in self.variables:
            self.mean_dict[col] = X[col].mean()
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col].fillna(self.mean_dict[col],inplace=True)
        return X

np.random.seed(0)
df = pd.DataFrame(np.random.randint(0,100,(10,2)),columns=["A",'B'])
df.iloc[1,0] = np.nan
df.iloc[2,1] = np.nan
df.iloc[3,1] = np.nan
df.iloc[4,0] = np.nan
df

mean_imputer = MeanImputer(variables=["A",'B'])

mean_imputer.fit(df)

mean_imputer.mean_dict

df.mean()

mean_imputer.transform(df)

import numpy
numpy.__version__

import pandas as pd
pd.__version__

import joblib
joblib.__version__

import sklearn
sklearn.__version__

import scipy
scipy.__version__

import setuptools
setuptools.__version__

import wheel
wheel.__version__

import pytest
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.predict import generate_predictions

def single_prediction():
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[config.FEATURES][:1]
    result = generate_predictions(single_row)
    return result

single_prediction()
