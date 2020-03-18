import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.initializers import glorot_normal
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

number = LabelEncoder()

# Este código sigue las indicaciones del libro Hands on Machine Learning with Scikit Learn and Tensorflow

# DEA

housing = pd.read_csv('housing.csv')

housing_with_id = housing.reset_index()

house_value_mean = housing["median_house_value"].mean()
house_value_sd = housing["median_house_value"].std()

# Split and stratify categories

housing_with_id['income_cat'] = np.ceil(housing_with_id['median_income'] / 1.5)
housing_with_id['income_cat'].where(housing_with_id['income_cat'] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_with_id, housing_with_id['income_cat']):
    strat_train_set = housing_with_id.loc[train_index]
    strat_test_set = housing_with_id.loc[test_index]

# print (housing_with_id['income_cat'].value_counts()/len(housing_with_id))

for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

housing = strat_train_set.copy()

# ML Data preparation

# Remove outcome from features matrix
housing = strat_train_set.drop("median_house_value", axis=1)

housing_labels = strat_train_set["median_house_value"].copy()

# Impute null values with mean

# Numerical Features - remove categorical feature
housing_num = housing.drop('ocean_proximity', axis=1)

imputer = SimpleImputer(strategy="median")
imputer.fit(housing_num)

X = imputer.transform(housing_num)

# Categorical Features
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)

# One Hot Encoding

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))

# Normalization

sc = StandardScaler()
X = sc.fit_transform(X)

# Traning and Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X, housing_labels, test_size=0.2, random_state=2)


# Model composition. 9 features, use relu activation for positive linear and random initialization with Xavier. 2 hidden layers

model = Sequential()
model.add(Dense(500, activation="relu",input_dim=9))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mean_squared_error'])
model.fit(X_train, Y_train, epochs=20)

Y_pred=model.predict(X_test)

print (np.sqrt(mean_squared_error(Y_test,Y_pred)))