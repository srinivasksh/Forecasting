import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

## Import the training file into pandas
df_full = pd.read_csv("train.csv")
df = df_full[0:1000]

traindf=df[0:700] 
testdf=df[700:]

print("Loaded file!")

traindf_sample = traindf[0:20]

## Plot to understand seasonality
plt.plot(traindf_sample['date'], traindf_sample['sales'],color='blue',label='Sales')
plt.show()

## Plot for sample data
#traindf_st10 = traindf[(traindf.store==1) & (traindf.item == 1)]
plt.plot(traindf['date'], traindf['sales'],color='blue',label='Sales')
plt.plot(testdf['date'], testdf['sales'],color='red',label='Sales')
plt.show()

## Exonential Smoothing

fit2 = SimpleExpSmoothing(np.asarray(traindf['sales'])).fit(smoothing_level=0.6,optimized=False)

ess_sales = fit2.forecast(len(testdf))
plt.plot(traindf['date'], traindf['sales'],color='blue',label='Sales')
plt.plot(testdf['date'], testdf['sales'],color='red',label='Sales')
plt.plot(testdf['date'], ess_sales,color='green',label='Sales')
plt.legend(loc='best')
plt.show()

## Winter Holt Exonential Smoothing
wh_model = ExponentialSmoothing(np.asarray(traindf['sales']) ,seasonal_periods=15 ,trend='add', seasonal='add',).fit()
Est_sales_holt  = wh_model.forecast(len(testdf))
plt.plot(traindf['date'], traindf['sales'],color='blue',label='Sales')
plt.plot(testdf['date'], testdf['sales'],color='red',label='Sales')
plt.plot(testdf['date'], Est_sales_holt,color='green',label='Sales')
plt.legend(loc='best')
plt.show()
