## Importing all relevant libraries....
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics


## Reading the dataset...
df = pd.read_csv('cancer.csv')
df.head()


## Drop unwanted columns like "id" and "Unnamed: 32"......
df.drop(['id' , 'Unnamed: 32'] , axis = 1 , inplace = True)


## Now our target feature is "diagnosis" and others are independent features....
## In 'diagnosis' column: 'M' means - "Malignant" & 'B' means - "Benign".............

df['diagnosis'] = df['diagnosis'].map({'M':1 , 'B':0})


## Target feature will be...
y = df['diagnosis']
y = y.values.reshape(-1,1)

## Independent features are....
x = df.drop('diagnosis' , axis = 1)


## scaling the data.....
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)

x = scaler.transform(x)

## Creating the pickle file for "Scaler" created....
pickle.dump(scaler , open('scaler_cancer.pkl' , 'wb'))


## Applying train-test split...
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.30 , random_state = 55)


## We can apply Logistic Regression since its a simple classification problem....
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

## Fiiting the model....
logreg.fit(x_train , y_train)


y_test_pred = logreg.predict(x_test)

## Checking the accuracy....
print("The Model's Test accuarcy is :" , metrics.accuracy_score(y_test , y_test_pred)*100, "%")


## Generting the pickle file of created model....
pickle.dump(logreg , open('cancer.pkl' , 'wb'))

