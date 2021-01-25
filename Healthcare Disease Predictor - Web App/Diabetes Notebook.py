## Importing all relevant libraries....
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics


df = pd.read_csv('diabetes.csv')


## Checking for null entries....
df.isnull().sum()


## Target feature will be...
y = df['Outcome']
y = y.values.reshape(-1,1)

## Independent features are....
x = df.drop('Outcome' , axis = 1)


## scaling the data.....
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)

x = scaler.transform(x)

## Creating the pickle file for "Scaler" created....
pickle.dump(scaler , open('scaler_diabetes.pkl' , 'wb'))


## Applying train-test split...
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.20 , random_state = 55)


## We can apply Logistic Regression since its a simple classification problem....
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=5)

## Fiiting the model....
logreg.fit(x_train , y_train)

## Checking accuracy for train data....
print("The Model's accuarcy for Training Data is :" , logreg.score(x_train , y_train)*100, "%")


## Predicting test data.....
y_test_pred = logreg.predict(x_test)

## Checking the accuracy....
print("The Model's Test accuarcy is :" , metrics.accuracy_score(y_test , y_test_pred)*100 , "%")

## Generting the pickle file of created model....
pickle.dump(logreg , open('diabetes_LR_Model.pkl' , 'wb'))


## We can also try RANDOM FOREST REGRESSION....
from sklearn.ensemble import RandomForestClassifier

rf_reg = RandomForestClassifier(n_estimators=20 , random_state=0)

## Fiiting the model....
rf_reg.fit(x_train , y_train)


## Predicting test data.....
y_test_pred = rf_reg.predict(x_test)

## Checking the accuracy....
print("The Model's Test accuarcy is :" , metrics.accuracy_score(y_test , y_test_pred)*100, "%")

## Generting the pickle file of created model....
pickle.dump(rf_reg , open('diabetes_RF_Model.pkl' , 'wb'))

