import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#understanding data
data=pd.read_csv("AmesHousing.csv")
print(data.describe())
print(data.columns)
print(data.isnull().sum().to_string())
print(data.info())
#cleaning data
