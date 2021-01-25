import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST']) 
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        
        Target_School = str(request.form['Target_School'])
        NBDE_1  = int(request.form['NBDE_1'])
        NBDE_2  = int(request.form['NBDE_2'])
        ADAT  = str(request.form['ADAT'])
        TOEFL_Min  = int(request.form['TOEFL_Min'])
        Home_Edu  = str(request.form['Home_Edu'])
        USA_Edu  = str(request.form['USA_Edu'])
        USA_Assistantship  = str(request.form['USA_Assistantship'])
        USA_Shadowing  = str(request.form['USA_Shadowing'])
        USA_Conferences  = str(request.form['USA_Conferences'])
        USA_Research  = str(request.form['USA_Research'])
        Home_Assistantship  = str(request.form['Home_Assistantship'])
        Home_Shadowing  = str(request.form['Home_Shadowing'])
        GPA  = float(request.form['GPA'])
        
        ## Slicing required information from the dataset....
        df = pd.read_csv('university_data.csv')
        ##...............................
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
        if uni_data['TOEFL_Min'].isnull().values == False:
            if TOEFL_Min < uni_data['TOEFL_Min'].values:
                flag = 1
        
        if TOEFL_Min == 120:
            x = x + 0.1*5
        elif TOEFL_Min >= 110:
            x = x + 0.1*4
        elif TOEFL_Min >= 100:
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
            return render_template('index.html', prediction_text='Oops Sorry! You do not cross the minimum criteria (i.e. GPA and TOEFL score cut-off) to get into the university!')
        else:
            return render_template('result.html', prediction_text1= Target_School, 
                                   prediction_text2 = uni_data['Space'].values[0] ,
                                   prediction_text4 = uni_data['GPA_Info'].values[0] ,
                                   prediction_text5 = uni_data['Extra'].values[0] ,
                                   prediction_text6 = round(prob , 1) ,
                                   prediction_text3 = uni_data['LOE'].values[0])
    
    
if __name__ == "__main__":
    app.run(debug=True)
