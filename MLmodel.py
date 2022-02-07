import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("Ames_Housing_Sales.csv")
print(data.describe())
print(data.columns)
print(data.isnull().sum())