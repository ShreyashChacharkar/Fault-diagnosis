from flask import Flask, render_template, request, jsonify,url_for
from utlis import GrpahBuilder, modelselect, lime_local_explain, ShapExplain
import pandas as pd
import numpy as np
import plotly.express as px
#import important liberies
import lime
import lime.lime_tabular
import shap
shap.initjs()

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
import matplotlib
matplotlib.use('Agg')

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
        le = modelselect('Label encoder')
        ls = ast.literal_eval(input_data[0])
        df1 = pd.DataFrame(data=np.expand_dims(np.array(ls[1:]),axis=0), columns=df.columns[1:])
        sc_data = sc.transform(df1)
        prediction = model.predict(sc_data)[0]
        lime_local = lime_local_explain(df.iloc[:,1:], sc_data, model)
        sh = ShapExplain(model,df,sc_data)
        shap_local = sh.shap_local_explainer()
        shap_global = sh.shap_global_explainer(save_path="web-dashboard/static/shap_global_summary1_plot.png")
    return render_template('index.html', dataframe=df, result=prediction,data=lime_local, shap_local=shap_local,shap_global=shap_global[14:])


    
@app.route('/failuredetection')
def failureprediction():
    fig = gb.scatter_graph()
    return render_template('index.html', dataframe=None)

@app.route('/faliurediagnosis')
def notification():
    return render_template('index.html', content='notification')

@app.route('/comment')
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