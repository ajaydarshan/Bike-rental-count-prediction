import os
import sys
if(len(sys.argv) != 2):
    print("Usage : python BikeSharingdemand.py path\\to\\day.csv ")
    sys.exit(1)
train_file = sys.argv[1]
os.chdir(os.curdir)
os.getcwd()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from fancyimpute import KNN  
from scipy.stats import chi2_contingency

data = pd.read_csv("day.csv")

data.head()

#check for null
data.isnull().sum()

data.info()

data.dtypes

data.describe()

data.drop(['instant','dteday'],axis=1,inplace=True)

data.head()

#sns.barplot(x='season',y='cnt',hue='mnth',data=data,estimator=np.std)
sns.barplot(x='season',y='cnt',hue ='yr',data=data)

sns.barplot(x='mnth',y='cnt',data=data,hue='yr')

sns.barplot(x='weathersit',y='cnt',data=data,hue='yr')

sns.barplot(x='weekday',y='cnt',data=data,hue='yr')

#sns.countplot(x='',data=data)

sns.lmplot(x='temp',y='cnt',data=data,col = 'season',row = 'yr')

sns.lmplot(x='hum',y='cnt',data=data,col = 'season',row = 'yr')

sns.lmplot(x='windspeed',y='cnt',data=data,col = 'season',row = 'yr')

#casual and registered are dependent variables, so remove them from model development--> casual + registered => cnt
data.drop(['casual','registered'],axis=1,inplace=True)


# # Outlier analysis
data.head()

#df = data.copy()

#data = df.copy()

cnames = ["temp","atemp","hum","windspeed","cnt"]

fig,ax = plt.subplots(5,figsize = (12,20))
for i in range(0,len(cnames)):
    sns.boxplot(y=cnames[i],x="weekday",data=data,ax=ax[i])

#Detect and delete outliers from data
for i in cnames:
    q75, q25 = np.percentile(data[i], [75 ,25])
    iqr = q75 - q25

    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)

    #Replace with NA
    data.loc[data[i] < min,i] = np.nan
    data.loc[data[i] > max,i] = np.nan

#Calculate missing value
missing_val = pd.DataFrame(data.isnull().sum())

#Impute with KNN
data = pd.DataFrame(KNN(k = 3).fit_transform(data), columns = data.columns)


# # Feature selection
data_corr = data.loc[:,cnames]

#Set the width and height of the plot
f, ax = plt.subplots(figsize=(12, 7))

#Generate correlation matrix
corr = data_corr.corr()

#Plot using seaborn library
#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#            square=True, ax=ax)
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap='coolwarm',square = True,linewidths = 1,ax=ax,annot =True)

cat_names = ["season","yr","mnth","holiday","weekday","workingday","weathersit"]
#loop for chi square values
for i in cat_names:
    for j in cat_names:
        if(i != j):
            chi2, p, dof, ex = chi2_contingency(pd.crosstab(data[i], data[j]))
            if(p < 0.05):
                print(i,"and",j,"are dependants")

#Drop columns
data.drop(['atemp','workingday'],axis=1,inplace=True)

data.head()

data.shape


# # Sampling
# 
#Import Libraries for decision tree
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

data.head()

data.columns

#Divide data into train and test
X = data.values[:, 0:9]
Y = data.values[:,9]
#X= data[['season', 'yr', 'mnth', 'holiday', 'weekday','weathersit', 'temp', 'hum', 'windspeed', 'casual', 'registered']]
#Y = data['cnt']

x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)


# # Model

# ## Linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_train,y_train)

model.coef_

pred_LR = model.predict(x_test)

plt.scatter(y_test,pred_LR)

sns.distplot((y_test-pred_LR))

def mape(y_test,pred):
    return np.mean(np.abs((y_test-pred)/y_test))

rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(pred_LR)),2)))
rms

metrics.mean_absolute_error(y_test,pred_LR) #mas

metrics.mean_squared_error(y_test,pred_LR) #mse

np.sqrt(metrics.mean_squared_error(y_test,pred_LR)) #rms

metrics.explained_variance_score(y_test,pred_LR)

#mape = np.mean(np.abs((np.array(y_test) - np.array(pred_LR))/np.array(y_test)))
#mape
mape(y_test,pred_LR)


# ## Random forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

model.fit(x_train,y_train)

pred_RF = model.predict(x_test)

plt.scatter(y_test,pred_RF)

sns.distplot((y_test-pred_RF))

rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(pred_RF)),2)))
rms

metrics.mean_absolute_error(y_test,pred_RF) #mas

np.sqrt(metrics.mean_squared_error(y_test,pred_RF)) #rms

metrics.explained_variance_score(y_test,pred_RF)

#mape = np.mean(np.abs((np.array(y_test) - np.array(pred))/np.array(y_test)))
#mape
mape(y_test,pred_RF)

np.mean(np.abs((np.array(y_test) - np.array(pred_RF))/np.array(y_test)))


# # Sample input and output to excel file
#pd.DataFrame([y_test,pred_LR,pred_RF])
#pd.DataFrame
data = {'True values': y_test, 'Linear regression pred': pred_LR,'Random Forest pred': pred_RF}
sampleoutput = pd.DataFrame(data)

sampleoutput.to_csv("predicted_values.csv")

#pd.DataFrame(x_test,columns = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'weathersit', 'temp',
#       'hum', 'windspeed']).to_csv("test_data.csv")

