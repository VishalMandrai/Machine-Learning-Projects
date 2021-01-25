## Importing all relevant libraries....
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics


df = pd.read_csv('indian_liver_patient.csv')

#----------------------------------------------------------------------------
## In "Albumin_and_Globulin_Ratio" we have 4 NAN values. We can drop these entries....
df.dropna(axis = 0 , inplace = True)


## Our target feature is "Dataset"....
## Here, '1' means - "Person have Liver Desease"
## and, '2' means - "Person Don't have Liver Desease"

df['Dataset'] = np.where(df['Dataset'] == 2 , 0 , 1)


## Applying feature engineering to "Gender" column....
df['Gender'] = df['Gender'].map({"Male" : 1 , "Female" : 0})


## Target feature will be...
y = df['Dataset']
y = y.values.reshape(-1,1)

## Independent features are....
x = df.drop('Dataset' , axis = 1)

#----------------------------------------------------------------------------
## scaling the data.....
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)

x = scaler.transform(x)

## Creating the pickle file for "Scaler" created....
pickle.dump(scaler , open('scaler_liver.pkl' , 'wb'))
#----------------------------------------------------------------------------

## Applying train-test split...
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.20 , random_state = 55)

#----------------------------------------------------------------------------

## We can use Gradient Boosting Technique as Classifier....
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(n_estimators=18)

## Fiiting the model....
gb_model.fit(x_train , y_train)

## Predicting test data.....
y_test_pred = gb_model.predict(x_test)

## Checking the accuracy....
print("The Model's Test accuarcy is :" , metrics.accuracy_score(y_test , y_test_pred)*100, "%")

## Generting the pickle file of created model....
pickle.dump(gb_model , open('liver.pkl' , 'wb'))
#----------------------------------------------------------------------------
