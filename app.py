from flask import Flask,request,render_template
from keras.models import load_model
import  pickle
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import tensorflow as tf



# loading models
app = Flask(__name__)
model = load_model("model.keras")
#scaler = pickle.load(open('scaler.pkl','rb'))

def pred(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    features = np.array([CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary])
    features = features.reshape(1, -1)  # Reshape to (1, input_dim)
    result = model.predict(features)
    #result = (result > 0.5).astype(int)
    return result[0]

# creating routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/churn_pred",methods=['POST','GET'])
def churn_pred():
    if request.method=='POST':
       CreditScore = int(request.form['CreditScore'])
       Geography = int(request.form['Geography'])
       Gender = int(request.form['Gender'])
       Age = int(request.form['Age'])
       Tenure = int(request.form['Tenure'])
       Balance = float(request.form['Balance'])
       NumOfProducts = int(request.form['NumOfProducts'])
       HasCrCard = int(request.form['HasCrCard'])
       IsActiveMember = int(request.form['IsActiveMember'])
       EstimatedSalary = float(request.form['EstimatedSalary'])

       prediction = pred(CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember, EstimatedSalary)
       prediction_text = "The Customer will Churn" if prediction == 1 else "The Customer will not Churn"
       return render_template('index.html',prediction=prediction_text)



if __name__ == "__main__":
    app.run(debug=True)