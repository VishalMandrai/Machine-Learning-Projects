import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index_main.html')


@app.route('/predict',methods=['POST']) 
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        reason = int(request.form['reason'])
        transportation_expense = int(request.form['transportation_expense'])
        distance_to_work = int(request.form['distance_to_work'])
        age = int(request.form['age'])
        daily_work_load_average = int(request.form['daily_work_load_average'])
        bmi = float(request.form['bmi'])
        education = float(request.form['education'])
        children = int(request.form['children'])
        pets = int(request.form['pets'])
        weekday = int(request.form['weekday'])
        
        ##--------------------------------------------------------------
        if reason >= 0 and reason <= 14:
            reasons_cate_1 = 1
        else:
            reasons_cate_1 = 0   
            
        if reason >= 15 and reason <= 17:
            reasons_cate_2 = 1
        else:
            reasons_cate_2 = 0   
        
        if reason >= 18 and reason <= 21:
            reasons_cate_3 = 1
        else:
            reasons_cate_3 = 0 
            
        if reason >= 22 and reason <= 28:
            reasons_cate_4 = 1
        else:
            reasons_cate_4 = 0 
        ##--------------------------------------------------------------
        
        data = np.array([[reasons_cate_1, reasons_cate_2, reasons_cate_3, reasons_cate_4, transportation_expense 
                          , distance_to_work , age , daily_work_load_average , bmi , education , children , pets , 
                          weekday]])
    
        scaler = pickle.load(open('scaler_absenteeism.pkl' , 'rb'))
        scaled_data = scaler.transform(data)
        
        
        logreg = pickle.load(open('model_LR.pkl', 'rb'))
        prediction = logreg.predict(scaled_data)
        
        
        if prediction == 1: 
            return render_template('index_main.html', prediction_text='Oops! Employee is likely to be absent for more than 3 Hours!')
        else:
            return render_template('index_main.html', prediction_text='Great! Your employee will be there for work!')
    
    
    
if __name__ == "__main__":
    app.run(debug=True)
