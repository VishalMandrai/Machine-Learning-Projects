## Importing relevant libraries....
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index_main.html')

##----------------------------------------------------------------------------
                             ## CANCER ##
##----------------------------------------------------------------------------
@app.route("/cancer")
def cancer():
    return render_template("cancer.html")

@app.route('/predict_cancer',methods=['POST'])
def predict_cancer():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]    
    final_features = [np.array(int_features)]
    
    ## Un-pickling scaler file....
    scaler = pickle.load(open("scaler_cancer.pkl" , "rb"))
        
    ## Un-pickling scaler file....
    logreg = pickle.load(open("cancer.pkl" , "rb"))
    
    temp = scaler.transform(final_features)
    prediction = logreg.predict(temp)

    if prediction == 1:
            return render_template('cancer.html', prediction_text='Oops SORRY! You have Cancer!')
    else:
        return render_template('cancer.html', prediction_text='Great News! You are healthy!')

##----------------------------------------------------------------------------
                             ## Diabetes ##
##----------------------------------------------------------------------------
                             
@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route('/predict_diabetes',methods=['POST'])
def predict_diabetes():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]    
    final_features = [np.array(int_features)]
    
    ## Un-pickling scaler file....
    scaler = pickle.load(open("scaler_diabetes.pkl" , "rb"))
        
    ## Un-pickling scaler file....
    RF_Model = pickle.load(open("diabetes_RF_Model.pkl" , "rb"))
    
    temp = scaler.transform(final_features)
    prediction = RF_Model.predict(temp)

    if prediction == 1:
            return render_template('diabetes.html', prediction_text='Oops SORRY! You have Diabetes!')
    else:
        return render_template('diabetes.html', prediction_text='Great News! You are healthy!')
    
##----------------------------------------------------------------------------
                             ## Heart ##
##----------------------------------------------------------------------------
                             
@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route('/predict_heart',methods=['POST'])
def predict_heart():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]    
    final_features = [np.array(int_features)]
    
    ## Un-pickling scaler file....
    scaler = pickle.load(open("scaler_heart.pkl" , "rb"))
        
    ## Un-pickling scaler file....
    logreg = pickle.load(open("heart.pkl" , "rb"))
    
    temp = scaler.transform(final_features)
    prediction = logreg.predict(temp)

    if prediction == 1:
            return render_template('heart.html', prediction_text='Oops SORRY! You have Heart Disease!')
    else:
        return render_template('heart.html', prediction_text='Great News! You are healthy!')

##----------------------------------------------------------------------------
                             ## Kidney ##
##----------------------------------------------------------------------------
                             
@app.route("/kidney")
def kidney():
    return render_template("kidney.html")

@app.route('/predict_kidney',methods=['POST'])
def predict_kidney():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]    
    final_features = [np.array(int_features)]
        
    ## Un-pickling scaler file....
    logreg = pickle.load(open("kidney.pkl" , "rb"))
    prediction = logreg.predict(final_features)

    if prediction == 1:
            return render_template('kidney.html', prediction_text='Oops SORRY! You have Kidney Disease!')
    else:
        return render_template('kidney.html', prediction_text='Great News! You are healthy!')

##----------------------------------------------------------------------------
                             ## Liver ##
##----------------------------------------------------------------------------
                             
@app.route("/liver")
def liver():
    return render_template("liver.html")

@app.route('/predict_liver',methods=['POST'])
def predict_liver():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]    
    final_features = [np.array(int_features)]
        
    ## Un-pickling scaler file....
    scaler = pickle.load(open("scaler_liver.pkl" , "rb"))
        
    ## Un-pickling scaler file....
    logreg = pickle.load(open("liver.pkl" , "rb"))
    
    temp = scaler.transform(final_features)
    prediction = logreg.predict(temp)

    if prediction == 1:
            return render_template('liver.html', prediction_text='Oops SORRY! You have Liver Disease!')
    else:
        return render_template('liver.html', prediction_text='Great News! You are healthy!')
    
##----------------------------------------------------------------------------
##----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)    