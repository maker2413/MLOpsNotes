#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

raw_train = pd.read_csv("train.csv")
raw_test = pd.read_csv("test.csv")

train_df = raw_train.copy()
test_df = raw_test.copy()

train_df.info()

test_df.info()
