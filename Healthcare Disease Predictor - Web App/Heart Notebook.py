## Importing all relevant libraries....
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics


df = pd.read_csv('heart.csv')
df.head()


## Checking for null entries....
df.isnull().sum()


## Target feature will be...
y = df['target']
y = y.values.reshape(-1,1)

## Independent features are....
x = df.drop('target' , axis = 1)


## scaling the data.....
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)

x = scaler.transform(x)

## Creating the pickle file for "Scaler" created....
pickle.dump(scaler , open('scaler_heart.pkl' , 'wb'))


## Applying train-test split...
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.20 , random_state = 55)


## We can use RANDOM FOREST REGRESSION as Classifier....
from sklearn.ensemble import RandomForestClassifier

rf_reg = RandomForestClassifier(n_estimators=100 , random_state=100)

## Fiiting the model....
rf_reg.fit(x_train , y_train)


## Predicting test data.....
y_test_pred = rf_reg.predict(x_test)

## Checking the accuracy....
print("The Model's Test accuarcy is :" , metrics.accuracy_score(y_test , y_test_pred)*100, "%")

## Generting the pickle file of created model....
pickle.dump(rf_reg , open('heart.pkl' , 'wb'))

