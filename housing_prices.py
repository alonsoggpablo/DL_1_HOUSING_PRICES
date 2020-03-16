import pandas as pd

# DEA
housing=pd.read_csv('housing.csv')

housing.info()

print (housing["ocean_proximity"].value_counts())


