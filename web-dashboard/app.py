from flask import Flask, render_template, request, jsonify,url_for
from utlis import GrpahBuilder, modelselect
import pandas as pd
import numpy as np
import plotly.express as px
#import important liberies

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, train_test_split

# # Import the required libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import ast


app = Flask(__name__)


df = pd.read_csv("dataset\dashboard.csv")
gb = GrpahBuilder()

@app.route('/')
@app.route('/rulprediction')
def rulprediction():
    return render_template('index.html',data="data", dataframe=df)

@app.route('/predict', methods=['GET', 'POST'])
def make_prediction():
    smodel = str(request.form["model"])
    input_data = request.form.getlist("input_data")
    if smodel=="None":
        prediction=False
    elif len(input_data[0])==0:
        prediction=False
    else:
        model = modelselect(smodel)
        sc = modelselect('Standard scalar')
        ls = ast.literal_eval(input_data[0])
        sc_data = sc.transform([ls[1:]])
        prediction = model.predict(sc_data)[0]
    return render_template('index.html',data="data", dataframe=df, result=prediction)
    
@app.route('/radio', methods=['POST'])
def radio_form():
    selected_option = request.form.get('options')

    predict_url = url_for('make_prediction')
    return render_template('index.html', data="data", dataframe=df, option=selected_option, predict_url=predict_url)

@app.route('/failureprediction')
def failureprediction():
    fig = gb.scatter_graph()
    return render_template('index.html', dataframe=None)

@app.route('/notification')
def notification():
    return render_template('index.html', content='notification')

@app.route('/#comment')
def comments():
    return render_template('index.html', content='comment')

@app.route('/#help')
def help():
    return render_template('index.html', content='help')

@app.route('/#share')
def share():
    return render_template('index.html', content='share')

if __name__ == '__main__':
    app.run(debug=True)