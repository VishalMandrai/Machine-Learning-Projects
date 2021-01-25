## Importing all relevant libraries....
import pandas as pd
import numpy as np
import pickle
import random
from sklearn import metrics


df = pd.read_csv('kidney_disease.csv')

## Applying Frequency impututation technique to deal with NAN entries............
df['bp'].fillna(df['bp'].value_counts().index[0] , inplace = True)
df['sg'].fillna(df['sg'].value_counts().index[0] , inplace = True)
df['al'].fillna(df['al'].value_counts().index[0] , inplace = True)
df['su'].fillna(df['su'].value_counts().index[0] , inplace = True)
df['pcc'].fillna(df['pcc'].value_counts().index[0] , inplace = True)
df['ba'].fillna(df['ba'].value_counts().index[0] , inplace = True)
df['htn'].fillna(df['htn'].value_counts().index[0] , inplace = True)
df['dm'].fillna(df['dm'].value_counts().index[0] , inplace = True)
df['cad'].fillna(df['cad'].value_counts().index[0] , inplace = True)
df['appet'].fillna(df['appet'].value_counts().index[0] , inplace = True)
df['pe'].fillna(df['pe'].value_counts().index[0] , inplace = True)
df['ane'].fillna(df['ane'].value_counts().index[0] , inplace = True)


## Creating a list of random samples from filled age entries....
temp = pd.Series(random.choices(df['age'].value_counts().index , k = df['age'].isnull().sum()))
temp.index = df[df['age'].isnull()].age.index
df.loc[df['age'].isnull() , 'age'] = temp

## Creating a list of random samples from filled 'rbc' entries....
temp1 = pd.Series(random.choices(['normal' , 'abnormal'], weights=(4,1), k=df['rbc'].isnull().sum()))
temp1.index = df[df['rbc'].isnull()].index
df.loc[df['rbc'].isnull() , 'rbc'] = temp1

## Creating a list of random samples from filled 'rbc' entries....
temp2 = pd.Series(random.choices(['normal' , 'abnormal'], weights=(5,1.5), k=df['pc'].isnull().sum()))
temp2.index = df[df['pc'].isnull()].index
df.loc[df['pc'].isnull() , 'pc'] = temp2

## Creating a list of random samples from filled 'bgr' entries....
temp3 = pd.Series(random.choices(df['bgr'].value_counts().index[0:5], k=df['bgr'].isnull().sum()))
temp3.index = df[df['bgr'].isnull()].index
df.loc[df['bgr'].isnull() , 'bgr'] = temp3

## Creating a list of random samples from filled 'bu' entries....
temp4 = pd.Series(random.choices(df['bu'].value_counts().index[0:5], k=df['bu'].isnull().sum()))
temp4.index = df[df['bu'].isnull()].index
df.loc[df['bu'].isnull() , 'bu'] = temp4

## Creating a list of random samples from filled 'sc' entries....
temp5 = pd.Series(random.choices(df['sc'].value_counts().index[0:5], k=df['sc'].isnull().sum()))
temp5.index = df[df['sc'].isnull()].index
df.loc[df['sc'].isnull() , 'sc'] = temp5

## Creating a list of random samples from filled 'sod' entries....
temp6 = pd.Series(random.choices(df['sod'].value_counts().index[0:13], k=df['sod'].isnull().sum()))
temp6.index = df[df['sod'].isnull()].index
df.loc[df['sod'].isnull() , 'sod'] = temp6

## Creating a list of random samples from filled 'pot' entries....
temp7 = pd.Series(random.choices(df['pot'].value_counts().index[0:5], k=df['pot'].isnull().sum()))
temp7.index = df[df['pot'].isnull()].index
df.loc[df['pot'].isnull() , 'pot'] = temp7

## Creating a list of random samples from filled 'hemo' entries....
temp7 = pd.Series(random.choices(df['hemo'].value_counts().index[0:15], k=df['hemo'].isnull().sum()))
temp7.index = df[df['hemo'].isnull()].index
df.loc[df['hemo'].isnull() , 'hemo'] = temp7

## Creating a list of random samples from filled 'pcv' entries....
temp8 = pd.Series(random.choices(df['pcv'].value_counts().index[0:6], k=df['pcv'].isnull().sum()))
temp8.index = df[df['pcv'].isnull()].index
df.loc[df['pcv'].isnull() , 'pcv'] = temp8


## Dropping "wc", "rc" and "id" features....
df.drop(['wc' , 'rc' , 'id'] , axis = 1 , inplace = True)


## Mapping categorical features to numeric values....
df['rbc'] = df['rbc'].map({'normal':1 , 'abnormal':0})
df['pc'] = df['pc'].map({'normal':1 , 'abnormal':0})
df['pcc'] = df['pcc'].map({'notpresent':1 , 'present':0})
df['ba'] = df['ba'].map({'notpresent':1 , 'present':0})
df['htn'] = df['htn'].map({'no':1 , 'yes':0})
df['dm'] = df['dm'].map({'no':1 , 'yes':0 , ' yes':0, '\tno':1, '\tyes':0})
df['cad'] = df['cad'].map({'no':1 , 'yes':0 , '\tno':1})
df['appet'] = df['appet'].map({'good':1 , 'poor':0})
df['pe'] = df['pe'].map({'no':1 , 'yes':0})
df['ane'] = df['ane'].map({'no':1 , 'yes':0})
df['classification'] = df['classification'].map({'ckd':1 , 'notckd':0 , 'ckd\t':1})


## Correcting some entries in "pcv" feature...
df['pcv'].replace({'\t43':43 , '\t?':df['pcv'].value_counts().index[0]} , inplace = True)
df['pcv'] = df['pcv'].astype('int64')


## Target feature will be...
y = df['classification']
y = y.values.reshape(-1,1)

## Independent features are....
x = df.drop('classification' , axis = 1)


## Applying train-test split...
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.30 , random_state = 55)


## Creating Model....
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
pickle.dump(logreg , open('kidney.pkl' , 'wb'))

