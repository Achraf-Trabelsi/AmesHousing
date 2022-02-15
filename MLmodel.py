import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

data=pd.read_csv("AmesHousing.csv")
#understanding data

print(data.shape)
print(data.columns)
missing_values=data.isnull().sum().to_string()
print(missing_values)
col_missing=[col for col in data.columns if data[col].isnull().sum()>0 ]
print(col_missing)

#cleaning data

#dropping unuseful data (Total already gives as the info)
data=data.drop(labels=['PID'],axis=1)
data=data.drop(labels=['BsmtFin SF 2', 'Bsmt Unf SF','BsmtFin SF 1'],axis=1)

#droping because number missing is so high
high_missing_rate=['Pool QC','Fence','Misc Feature','Alley','Fireplace Qu']
data=data.drop(labels=high_missing_rate,axis=1)

#imputing for categorical features
cat_fea=['Garage Type','Garage Finish','Garage Qual',
'Garage Cond','Lot Frontage','Bsmt Qual', 'Bsmt Cond', 
'Bsmt Exposure','BsmtFin Type 1', 'BsmtFin Type 2',
'Mas Vnr Type','Electrical']
for fea in cat_fea:
    data[fea].fillna(data[fea].mode()[0],inplace=True)
#imputing for numerical features
num_fea=['Bsmt Full Bath','Bsmt Half Bath','Total Bsmt SF']
for fea in num_fea:
    data[fea].fillna(data[fea].mean(),inplace=True)

data['Garage Yr Blt'].fillna(value=data['Garage Yr Blt'].median(),inplace=True)
data['Garage Cars'].fillna(value=0,inplace=True)
data['Garage Area'].fillna(value=0,inplace=True)
data['Mas Vnr Area'].fillna(value=data['Mas Vnr Area'].median(),inplace=True)
#visualization
corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

#train test 
target=data.SalePrice
d=data.drop(labels='SalePrice',axis=1)
dtrain,dtest,ptrain,ptest=train_test_split(d,target,train_size=0.8,random_state=True)