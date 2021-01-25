## importing relevant libraries....
import pandas as pd
import numpy as np
import random

import pickle
from sklearn import metrics


## Importing the dataset....
df = pd.read_csv("Absenteeism_data.csv")
df.head()


## dropping "ID" column.....
df.drop(['ID'] , axis = 1 , inplace = True)

## In "Resaons to Absence" there are 29 Different excuses....
## We need to condense the column into possible type as we can't create 29 New Columns....

reason_to = pd.get_dummies(df['Reason for Absence'] , drop_first = True)

## Here we collecting the columns of interest.......
## We creating the collection depends on the column name and creating a set 4 new features.....

rea_1 = [x for x in reason_to.columns.values if x<=14]
rea_2 = [x for x in reason_to.columns.values if x>=15 if x<=17]
rea_3 = [x for x in reason_to.columns.values if x>=18 if x<=21]
rea_4 = [x for x in reason_to.columns.values if x>=22 if x<=28]

## Creating corresponding new 4 features.....
reasons_cate_1 = reason_to[rea_1].max(axis = 1)
reasons_cate_2 = reason_to[rea_2].max(axis = 1)
reasons_cate_3 = reason_to[rea_3].max(axis = 1)
reasons_cate_4 = reason_to[rea_4].max(axis = 1)

## Creating the dataframe....
temp = pd.DataFrame({'reasons_cate_1':reasons_cate_1 , 'reasons_cate_2':reasons_cate_2 , 'reasons_cate_3':reasons_cate_3 , 
                     'reasons_cate_4':reasons_cate_4})

## Adding the new columns to main dataframe....
df = pd.concat([temp , df] , axis = 1)


## dropping "Reason for Absence	"....
df.drop('Reason for Absence' , axis = 1 , inplace = True)

## Now our Reasons Columns is fully processed....


## here we converting the Date Column to time stamp Format. so, that we can extract information...
## We'll be needing the weekday info. which we'll get from here.....
dates = pd.to_datetime(df['Date'] , format = '%d-%m-%Y')
df['Date'] = dates

## Creating the list of weekdays from the given dates....
list_of_weekdays = []

for i in range(df.shape[0]):
    list_of_weekdays.append(df['Date'][i].weekday())

## adding the created list to data frame....
df['Weekday'] = list_of_weekdays


## Dropping the "Date" Column.....
df.drop(['Date'] , axis = 1 , inplace = True)



# #### In Education Column:
# - 1 : High School Graduate
# - 2 : College Graduate
# - 3 : Post Graduate
# - 4 : PhD Qualified
# 
#         - So no need to further change the data. This is ordinal data and optimization is as per Ordinal Feature. 


## All entries should be less than '24'...
## Applying Random Sample Imputation for all entries above 24....
sample = pd.Series(random.choices(df['Absenteeism Time in Hours'].value_counts().head(3).index , weights = (1.5 , 1 , 1) , 
                        k = df[df['Absenteeism Time in Hours'] > 24].shape[0]))

sample.index = df[df['Absenteeism Time in Hours'] > 24].index
df.loc[df['Absenteeism Time in Hours'] > 24 , 'Absenteeism Time in Hours'] = sample


## Creating the "Targets" column for the data set,.......
df['target'] = np.where(df['Absenteeism Time in Hours'] > df['Absenteeism Time in Hours'].median() , 1 , 0)

## Dropping "Absence Time in Hours" column.............
df.drop(['Absenteeism Time in Hours'] , axis = 1 , inplace = True)


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
pickle.dump(scaler , open('scaler_absenteeism.pkl' , 'wb'))


## Applying train-test split...
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.1 , random_state = 55)


## We can apply Logistic Regression since its a simple classification problem....
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state = 0)

## Fiiting the model....
logreg.fit(x_train , y_train)

## Trainjng Accuracy of the model....
print(logreg.score(x_train , y_train))


y_test_pred = logreg.predict(x_test)

## Checking the accuracy....
print("The Model's Test accuarcy is :" , metrics.accuracy_score(y_test , y_test_pred)*100, "%")


## Generting the pickle file of created model....
pickle.dump(logreg , open('model_LR.pkl' , 'wb'))

