import warnings
warnings.filterwarnings('ignore')

# <--DATA UNDERSTANDING-->

# import libraries
import pandas as pd
import numpy as np
#reading the dataset
dataset = pd.read_csv("bike_sharing_data.csv")
#print(dataset.head())
#print(dataset.shape)
#print(dataset.columns)
#print(dataset.describe())
#print(dataset.info())


    # Assigning string values to different seasons instead of numeric values 
dataset['season'] = dataset['season'].replace({
    1: 'spring',
    2: 'summer',
    3: 'fall',
    4: 'winter'
})
    # 0 - 2018 1 - 2019
#print(dataset['yr'].astype('category').value_counts())

    # Assigning string values to different months instead of numeric values
def object_map_mnth(x):
    return x.map({1: 'jan', 2:'feb', 3:'mar',4:'apr',5:'may', 6:'jun',7:'jul',8:'aug', 9:'sept',10:'oct', 11:'nov',12:'dec' })
dataset[['mnth']]=dataset[['mnth']].apply(object_map_mnth)
#print(dataset['mnth'].astype('category').value_counts())

#print(dataset['holiday'].astype('category').value_counts())

    # assigning  string values to different weekdays instead of numeric values
def str_map_weekday(x):
    return x.map({1: 'Mon',2:'Thues',3:'Wed',4:'Thurs',5:'Fri',6:'Sat',7:'Sun'})
dataset[['weekday']]=dataset[['weekday']].apply(str_map_weekday)
#print(dataset['weekday'].astype('category').value_counts())

#print(dataset['workingday'].astype('category').value_counts())

dataset['weathersit'] = dataset['weathersit'].replace({
    1: 'A',# clear, few clouds , partly cloudy
    2: 'B',# mist, cloudy
    3: 'C'  #light snow, heavy rain
})
#print(dataset['weathersit'].astype('category').value_counts())

#  <--DATA VISUALISATION-->

import matplotlib.pyplot as plt
import seaborn as sns

    #Temperature
sns.displot(dataset['temp'])
#plt.show()

    #Actual Temperature
sns.displot(dataset['atemp'])
#plt.show()

    # wind speed
sns.displot(dataset['windspeed'])
#plt.show()

    # target variable: count of total rental bikes includes both casual and registered
sns.displot(dataset['cnt'])
#plt.show()

    # converting date to datetime format
dataset['dteday']= pd.to_datetime(dataset['dteday'],format='%d-%m-%Y')

dataset_categorial = dataset.select_dtypes(exclude=['float64', 'datetime64','int64'])
#print(dataset_categorial.columns)
#print(dataset_categorial)

plt.figure(figsize=(20,20))
plt.subplot(3,3,1)
sns.boxplot(x = 'season', y = 'cnt', data = dataset)
plt.subplot(3,3,2)
sns.boxplot(x = 'mnth', y = 'cnt', data = dataset)
plt.subplot(3,3,3)
sns.boxplot(x = 'weekday', y = 'cnt', data = dataset)
plt.subplot(3,3,4)
sns.boxplot(x = 'weathersit', y = 'cnt', data = dataset)
plt.subplot(3,3,5)
sns.boxplot(x = 'workingday', y = 'cnt', data = dataset)
plt.subplot(3,3,6)
sns.boxplot(x = 'holiday', y = 'cnt', data = dataset)
plt.subplot(3,3,7)
sns.boxplot(x = 'yr', y = 'cnt', data = dataset)

intVarlist = ["casual", "registered", "cnt"]
for var in intVarlist:
   dataset[var]= dataset[var].astype("float")
dataset_numeric= dataset.select_dtypes(include=['float64'])
#print(dataset_numeric.head())

sns.pairplot(dataset_numeric)
#plt.show()

cor = dataset_numeric.corr()
#print(cor)

    # heatmap
mask= np.array(cor)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
fig.set_size_inches(10,10)
sns.heatmap(cor, mask=mask, vmax=8, square= True, annot=True)
plt.show()

    # removing atemp as it is highly correlated with temp
dataset.drop('atemp', axis=1, inplace=True)
#print(dataset.head())

#  <-- Data Preparation-->

dataset_categorial = dataset.select_dtypes(include=['object'])
dataset_categorial.head()
   
    # creating dummies for above dataset_categorical
dataset_dummies = pd.get_dummies(dataset_categorial, drop_first = True)
dataset_dummies=dataset_dummies.astype(int)
#print(dataset_dummies.head())
    
    # Dropping Catergorical variable columns
dataset = dataset.drop(list(dataset_categorial.columns),axis=1)
#print(dataset.head())

    # Concatenate  dummy variable with dataset
dataset = pd.concat([dataset, dataset_dummies], axis=1)
#print(dataset.head())

    # removing insignificant dteday and instant
dataset = dataset.drop(['instant', 'dteday'], axis=1, inplace= False)

#  <-- MODEL BUILDING AND EVALUATION-->
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
    #split dataframe into train and test datasets
from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train, df_test = train_test_split(dataset, train_size=0.7 , test_size=0.3)

    # Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

    # Apply scalar to all columns except dummy variables
var =["temp", "hum", "windspeed", "casual","registered","cnt"]
df_train[var]= scaler.fit_transform(df_train[var])
#print(df_train.describe())
   
    #Checking the correlation coefficients to see which variabkes are highly correlated
plt.figure(figsize=(30,30))
sns.heatmap(df_train.corr(),annot= True, cmap="YlGnBu")
plt.show()

    # Diving into X and y while dropping registered and casual
y_train = df_train.pop('cnt')
X_train = df_train.drop(["casual", "registered"], axis=1)
#print(X_train.head())
np.array(X_train)
   
    # stats model
import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_lm).fit()
#print(lr.params)

lm= LinearRegression()
#lm.fit(X_train, y_train)
#print(lm.coef_)
#print(lm.intercept_)

    # summary of lr
#print(lr.summary())     USING ONLY 15 FEATURES TO ACQUIRE SIMILAR RESULTS RATHER THAN USING ALL FEATURES

    # importing recurrsive feature elimination -rfe
from sklearn.feature_selection import RFE
lm = LinearRegression()

   # fitting only with 15 features
rfe1 = RFE(estimator=lm,n_features_to_select= 15)
rfe1= rfe1.fit(X_train, y_train)
#print(rfe1.support_)
#print(rfe1.ranking_)
col1 = X_train.columns[rfe1.support_]
#print(col1)

X_train_rfe1= X_train[col1]
X_train_rfe1 = sm.add_constant(X_train_rfe1)
lm1= sm.OLS(y_train, X_train_rfe1).fit()
print(lm1.summary())

from statsmodels.stats.outliers_influence import variance_inflation_factor
a= X_train_rfe1.drop('const', axis=1)
    
    # Evaluating VIFs
vif = pd.DataFrame()
vif['features']= a.columns
vif['VIF']=[variance_inflation_factor(a.values, i) for i in range(a.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif = vif.sort_values(by= "VIF",ascending= False)
print(vif)

    # predicting values
y_train_cnt = lm1.predict(X_train_rfe1)
fig = plt.figure()
sns.displot((y_train, y_train_cnt),bins =20)

df_test[var] = scaler.transform(df_test[var])
#print(df_test.columns)
y_test = df_test.pop('cnt')
X_test = df_test.drop(["casual","registered"], axis=1)
X_test_rfe1=X_test[a.columns]
X_test_rfe1= sm.add_constant(X_test_rfe1)
y_pred = lm1.predict(X_test_rfe1)

plt.figure()
plt.scatter(y_test, y_pred)
plt.show()

    #printing performance scores
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

plt.figure(figsize=(8,5))
sns.heatmap(dataset[col1].corr() ,cmap = "YlGnBu",annot=True)
plt.show()
