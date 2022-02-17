from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
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
data=data.drop(labels=['PID','Order'],axis=1)
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

data['SalePrice'].hist(bins=50, figsize=(5, 5))
plt.show()
plt.xlabel('SalePrice')
log_price=np.log(data['SalePrice'])
log_price.hist(bins=50, figsize=(5, 5))
plt.xlabel('log price')
plt.show()

data['Lot Frontage'].hist(bins=50, figsize=(5, 5))
plt.xlabel('Lot Frontage')
plt.show()
log_Lot=np.log(data['Lot Frontage'])
log_Lot.hist(bins=50, figsize=(5, 5))
plt.xlabel('Lot log Frontage')
plt.show()

month=plt.scatter(data['Mo Sold'],data['SalePrice'])
plt.xlabel('Mo sold')
plt.ylabel('Sale Price')
plt.show()
year=plt.scatter(data['Yr Sold'],data['SalePrice'])
plt.xlabel('Yr sold')
plt.ylabel('Sale Price')
plt.show()
Lot=plt.scatter(data['Lot Frontage'],data['SalePrice'])
plt.xlabel('Lot Frontage')
plt.ylabel('Sale Price')
plt.show()
Lot=plt.scatter(log_Lot,data['SalePrice'])
plt.xlabel('Lot log Frontage')
plt.ylabel('Sale Price')
plt.show()
neighbor=plt.bar(data['Neighborhood'],data['SalePrice'])
plt.xlabel('Neighbor')
plt.ylabel('Sale Price')
plt.show()
#features are irrelevant
data=data.drop(labels=["Yr Sold","Mo Sold","Misc Val","Pool Area",
"MS SubClass","Overall Cond","Bsmt Half Bath","3Ssn Porch",
"Low Qual Fin SF","Low Qual Fin SF",'Bedroom AbvGr',
'Kitchen AbvGr','Enclosed Porch','Screen Porch'],axis=1)

corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

#normal distributions
data["Log Lot"]=log_Lot
data=data.drop(labels='Lot Frontage',axis=1)
data["log Price"]=log_price
print(data.columns)
print(data.shape)
#train test 
target=data.SalePrice
d=data.drop(labels='SalePrice',axis=1)
d_train,d_test,ptrain,ptest=train_test_split(d,target,train_size=0.8,random_state=True)

#data preprocessiong
enc=OrdinalEncoder()
d_train=enc.fit_transform(d_train)
d_test=enc.fit_transform(d_test)

#Linear regression

clf=LinearRegression()
clf.fit(d_train,ptrain)
clf.predict(d_test)


