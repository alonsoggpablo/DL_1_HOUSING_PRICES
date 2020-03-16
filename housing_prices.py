import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

# Este código sigue las indicaciones del libro Hands on Machine Learning with Scikit Learn and Tensorflow

# DEA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

housing=pd.read_csv('housing.csv')

housing.info()

print (housing["ocean_proximity"].value_counts())

print (housing.describe())

housing.hist(bins=50,figsize=(20,15))

plt.show()

housing_with_id=housing.reset_index()
train_set,test_set=train_test_split(housing_with_id,test_size=0.2,random_state=42)

# Split and stratify categories

housing_with_id['income_cat']=np.ceil(housing_with_id['median_income']/1.5)
housing_with_id['income_cat'].where(housing_with_id['income_cat']<5,5.0,inplace=True)

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing_with_id,housing_with_id['income_cat']):
    strat_train_set=housing_with_id.loc[train_index]
    strat_test_set=housing_with_id.loc[test_index]

# print (housing_with_id['income_cat'].value_counts()/len(housing_with_id))

for set_ in (strat_train_set,strat_test_set):
    set_.drop('income_cat',axis=1,inplace=True)

housing=strat_train_set.copy()

housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,
             s=housing['population']/100,label='population',figsize=(10,7),
             c='median_house_value',cmap=plt.get_cmap('jet'),colorbar=True)

# Correlations

attributes=['median_house_value','median_income','total_rooms','housing_median_age']
scatter_matrix(housing[attributes],figsize=(12,8))

housing.plot(kind='scatter',x='median_income',y='median_house_value')
plt.show()



housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
print (corr_matrix["median_house_value"].sort_values(ascending=False))

# ML Data preparation

# Remove outcome from features matrix
housing = strat_train_set.drop("median_house_value", axis=1)

housing_labels = strat_train_set["median_house_value"].copy()

# Impute null values with mean

# Numerical Features - remove categorical feature
housing_num=housing.drop('ocean_proximity',axis=1)

imputer = SimpleImputer(strategy="median")
imputer.fit(housing_num)

X=imputer.transform(housing_num)

# Categorical Features
encoder=LabelEncoder()
housing_cat=housing['ocean_proximity']
housing_cat_encoded=encoder.fit_transform(housing_cat)

# One Hot Encoding

encoder=OneHotEncoder()
housing_cat_1hot=encoder.fit_transform(housing_cat_encoded.reshape(-1,1))

# Normalization

sc=StandardScaler()
X=sc.fit_transform(X)

# Traning and Evaluating

lin_reg=
