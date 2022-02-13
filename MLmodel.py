from cProfile import label
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split

#understanding data
data=pd.read_csv("AmesHousing.csv")
#print(data.describe())
print(data.shape)
print(data.columns)
missing_values=data.isnull().sum().to_string()

#print(data.info())
#cleaning data
#droping 
data=data.drop(labels='Pool QC',axis=1)
data=data.drop(labels='Fence',axis=1)
data=data.drop(labels='Misc Feature',axis=1)
data=data.drop(labels='Alley',axis=1)
data=data.drop(labels='Fireplace Qu',axis=1)
#imputing
data['Garage Type'].fillna(data['Garage Type'].mode()[0], inplace = True)
data['Garage Finish'].fillna(data['Garage Finish'].mode()[0], inplace = True)
data['Garage Qual'].fillna(data['Garage Qual'].mode()[0], inplace = True)
data['Garage Cond'].fillna(data['Garage Cond'].mode()[0], inplace = True)
data['Lot Frontage'].fillna(data['Lot Frontage'].mode()[0], inplace = True)
print(data.isnull().sum().to_string())

#train test 
target=data.SalePrice
d=data.drop(labels='SalePrice',axis=1)
dtrain,dtest,ptrain,ptest=train_test_split(d,target,train_size=0.8,random_state=True)