## Importing all required libraries...
import numpy as np
import pandas as pd

df = pd.read_csv('university_data.csv')

## Our input variables....
Target_School = 'Virginia Commonwealth University School of Dentistry'

NBDE_1 = 1
NBDE_2 = 1
ADAT = 'Yes'
TOEFL_Min = 120
Home_Edu = 'MDS'
USA_Edu = 'Residency'
USA_Assistantship = '>200 hours'
USA_Shadowing = '>200 hours'
USA_Conferences = 'Presented'
USA_Research = 'Systematic review'
Home_Assistantship = '>200 hours'
Home_Shadowing = '>200 hours'
GPA = 9.9


## Slicing required information from the dataset....
uni_data = df[df.University_Name == Target_School] 



#---------------------------------------------------------------------------------
## Initialising the container variable....
x = 0

## Total propability......
Contribution_of_NBDE_1 = 0.05
Contribution_of_NBDE_2 = 0.05
Contribution_of_ADAT = 0.05
Contribution_of_TOEFL_Min = 0.1
Contribution_of_Home_Edu = 0.05
Contribution_of_USA_Edu = 0.1
Contribution_of_USA_Assistantship = 0.125
Contribution_of_USA_Shadowing = 0.125
Contribution_of_USA_Conferences = 0.1
Contribution_of_USA_Research = 0.1
Contribution_of_Home_Assistantship = 0.075
Contribution_of_Home_Shadowing = 0.075

y = (Contribution_of_NBDE_1*1 + Contribution_of_NBDE_2*1 + Contribution_of_ADAT*3 + Contribution_of_TOEFL_Min*5 + 
     Contribution_of_Home_Edu*5 + Contribution_of_USA_Edu*5 + Contribution_of_USA_Assistantship*5 + 
     Contribution_of_USA_Shadowing*5 + Contribution_of_USA_Conferences*5 + Contribution_of_USA_Research*5 + 
     Contribution_of_Home_Assistantship*5 + Contribution_of_Home_Shadowing*5) 

## Creating flag for including exceptional conditions........
flag = 0
#---------------------------------------------------------------------------------



##Computing for "NBDE 1" scores..........
if NBDE_1 == 1:
    x = x + 1*0.05
elif NBDE_1 == 2:
    x = x + -0.1*0.05
elif NBDE_1 == 3:
    x = x + -0.2*0.05
elif NBDE_1 == 4:
    x = x + -0.3*0.05
elif NBDE_1 == 5:
    x = x + -0.5*0.05



##Computing for "NBDE 2" scores..........
if NBDE_2 == 1:
    x = x + 1*0.05
elif NBDE_2 == 2:
    x = x + -0.1*0.05
elif NBDE_2 == 3:
    x = x + -0.2*0.05
elif NBDE_2 == 4:
    x = x + -0.3*0.05
elif NBDE_2 == 5:
    x = x + -0.5*0.05


##Computing for "ADAT"..........
if ADAT == 'Yes':
    x = x + 0.05*3


##Computing for "TOEFL"..........
if uni_data['TOEFL_Min'].values.isnull() == False:
    if TOEFL < uni_data['TOEFL_Min'].values:
    flag = 1

if TOEFL == 120:
    x = x + 0.1*5
elif TOEFL >= 110:
    x = x + 0.1*4
elif TOEFL >= 100:
    x = x + 0.1*3
else:
    x = x + 0.1*2

    
    
##Computing for "Home Education"..........
if Home_Edu == 'MDS':
    x = x + 0.05*5
elif Home_Edu == 'Other':
    x = x + 0.05*3


##Computing for "USA Education"..........
if USA_Edu == 'Residency':
    x = x + 0.1*5
elif USA_Edu == 'PhD':
    x = x + 0.1*5
elif USA_Edu == 'Fellowship':
    x = x + 0.1*4
elif USA_Edu == 'MS':
    x = x + 0.1*4
elif USA_Edu == 'MHA':
    x = x + 0.1*3
elif USA_Edu == 'MPH':
    x = x + 0.1*3
elif USA_Edu == 'Certifications (not CE)':
    x = x + 0.1*2
elif USA_Edu == 'Other':
    x = x + 0.1*1


##Computing for "USA Assistantship"..........
if USA_Assistantship == '<150 hours':
    x = x + 0.125*-1
elif USA_Assistantship == '>200 hours':
    x = x + 0.125*5
elif USA_Assistantship == '<200 hours':
    x = x + 0.125*3


##Computing for "USA Shadowing"..........
if USA_Shadowing == '<150 hours':
    x = x + 0.125*-1
elif USA_Shadowing == '>200 hours':
    x = x + 0.125*5
elif USA_Shadowing == '<200 hours':
    x = x + 0.125*3


##Computing for "USA Conference"..........
if USA_Conferences == 'Presented':
    x = x + 0.1*5
elif USA_Conferences == 'Attended':
    x = x + 0.1*3


##Computing for "USA Research"..........
if USA_Research == 'Systematic review':
    x = x + 0.1*5
elif USA_Research == 'Clinical trials':
    x = x + 0.1*4
elif USA_Research == 'Case reports':
    x = x + 0.1*3
elif USA_Research == 'Literature review':
    x = x + 0.1*2
elif USA_Research == 'Research assistant':
    x = x + 0.1*1


##Computing for "Home Assistanship"..........
if Home_Assistantship == '<150 hours':
    x = x + 0.075*-1
elif Home_Assistantship == '>200 hours':
    x = x + 0.075*5
elif Home_Assistantship == '<200 hours':
    x = x + 0.075*3


##Computing for "Home Shadowing"..........
if Home_Shadowing == '<150 hours':
    x = x + 0.075*-1
elif Home_Shadowing == '>200 hours':
    x = x + 0.075*5
elif Home_Shadowing == '<200 hours':
    x = x + 0.075*3

## Computing for "Minimum GPA requirement"..........
if GPA < uni_data['GPA_Min'].values:
    flag = 1


## Total chances of admit are.....
prob = (x*100)/y


## Final result output stage....
if flag == 1:
    print('You do not cross the minimum criteria (GPA and TOEFL score) to get into the university!')
else:
    print('YOUR TARGET SCHOOL:', uni_data['University_Name'].values[0])
    print('CLASS SIZE:', uni_data['Space'].values[0])
    print('LETTERS OF EVALUATION REQUIRED: ', uni_data['LOE'].values[0])
    print('INFORMATION AVAILABLE ON GPA: ', uni_data['GPA_Info'].values[0])
    print('ANY ADDITIONAL INFORMATION AVAILABLE: ', uni_data['Extra'].values[0])
    print('YOUR PROBABILITY OF GETTING SELECTED TO AN INTERVIEW WITH THE SCHOOL:', round(prob , 1), '%')
        
