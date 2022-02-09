import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import numpy as np

#understanding data
data=pd.read_csv("AmesHousing.csv")
#print(data.describe())
print(data.shape)
print(data.columns)
missing_values=data.isnull().sum().to_string()
print(missing_values)
#print(data.info())
#cleaning data
#droping 
data=data.drop(labels='Pool QC',axis=1)
data=data.drop(labels='Fence',axis=1)
data=data.drop(labels='Misc Feature',axis=1)
data=data.drop(labels='Alley',axis=1)
#imputing
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data=imp.fit_transform(data)
