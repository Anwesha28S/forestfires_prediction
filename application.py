import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler




from flask import Flask,jsonify,request,render_template
application=Flask(__name__)
app=application
ridge_model=pickle.load(open("models/ridge.pkl","rb"))
standard_scaler=pickle.load(open("models/scaler.pkl","rb"))
@app.route("/")
def index():
    return render_template('index.html')
@app.route("/predictdata",methods=["GET","POST"])
def predict_data():
    if request.method=="POST":
        # Fetch data from form
        Temperature = float(request.form['temperature'])
        RH = float(request.form['rh'])
        WS = float(request.form['ws'])
        RAIN = float(request.form['rain'])
        FMMC = float(request.form['fmmc'])
        DMC = float(request.form['dmc'])
        ISI = float(request.form['isi'])
        
        # classes and region (categorical)
        classes = float(request.form['classes'])
        region = float(request.form['region'])
        new_data=standard_scaler.transform([[Temperature,RH,WS,RAIN,FMMC,DMC,ISI,classes,region]])
        result=ridge_model.predict(new_data)
        return render_template("home.html",results=result[0])
    else:
        return render_template("home.html",results=None)
if(__name__)=="__main__":
    app.run(host="0.0.0.0",debug=True)