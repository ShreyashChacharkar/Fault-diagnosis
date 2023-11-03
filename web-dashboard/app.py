from flask import Flask, render_template, request
from utlis import GrpahBuilder
import pandas as pd
import numpy as np
import plotly.express as px

app = Flask(__name__)



gb = GrpahBuilder()

@app.route('/')
@app.route('/rulprediction')
def home():
    fig = gb.graph()
    return render_template('index.html', result = fig.to_html())

@app.route('/failureprediction')
def failureprediction():
    fig = gb.scatter_graph()
    return render_template('index.html',  result = fig.to_html())

@app.route('/#notification')
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