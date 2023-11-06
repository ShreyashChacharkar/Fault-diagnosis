
import pandas as pd
import numpy as np
import plotly.express as px
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

import plotly.graph_objects as go

import joblib




class GrpahBuilder:
    def __init__(self) -> None:
        self.date_rng = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        self.data = {
            "Date": self.date_rng,
            "ActualValues": np.sin(np.linspace(0, 2 * np.pi, num=len(self.date_rng))),
            "PredictedValues": np.sin(np.linspace(0, 2 * np.pi, num=len(self.date_rng))) + np.random.normal(0, 0.1, len(self.date_rng)),
        }
        self.df = pd.DataFrame(self.data)

    def graph(self):
        np.random.seed(1)

        x0 = np.random.normal(2, 0.4, 400)
        y0 = np.random.normal(2, 0.4, 400)
        x1 = np.random.normal(3, 0.6, 600)
        y1 = np.random.normal(6, 0.4, 400)
        x2 = np.random.normal(4, 0.2, 200)
        y2 = np.random.normal(4, 0.4, 200)

        fig = go.Figure()

# Add traces
        fig.add_trace(
            go.Scatter(
                x=x0,
                y=y0,
                mode="markers",
                marker=dict(color="DarkOrange")
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x1,
                y=y1,
                mode="markers",
                marker=dict(color="Crimson")
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x2,
                y=y2,
                mode="markers",
                marker=dict(color="RebeccaPurple")
            )
        )

# Add buttons that add shapes
        cluster0 = [dict(type="circle",
                                    xref="x", yref="y",
                                    x0=min(x0), y0=min(y0),
                                    x1=max(x0), y1=max(y0),
                                    line=dict(color="DarkOrange"))]
        cluster1 = [dict(type="circle",
                                    xref="x", yref="y",
                                    x0=min(x1), y0=min(y1),
                                    x1=max(x1), y1=max(y1),
                                    line=dict(color="Crimson"))]
        cluster2 = [dict(type="circle",
                                    xref="x", yref="y",
                                    x0=min(x2), y0=min(y2),
                                    x1=max(x2), y1=max(y2),
                                    line=dict(color="RebeccaPurple"))]

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="None",
                            method="relayout",
                            args=["shapes", []]),
                        dict(label="Cluster 0",
                            method="relayout",
                            args=["shapes", cluster0]),
                        dict(label="Cluster 1",
                            method="relayout",
                            args=["shapes", cluster1]),
                        dict(label="Cluster 2",
                            method="relayout",
                            args=["shapes", cluster2]),
                        dict(label="All",
                            method="relayout",
                            args=["shapes", cluster0 + cluster1 + cluster2])
                    ],
                )
            ]
        )

        # Update remaining layout properties
        fig.update_layout(
            title_text="Highlight Clusters",
            showlegend=False,
        )

        return fig
    
    def scatter_graph(self):
        data1  = {
                    'x': [1, 2, 3, 4, 5],
                    'y': [2, 4, 1, 3, 5]
                }
        fig = px.scatter(data1, x='x', y='y', title='Scatter Plot')
        return fig



def modelselect(model):
    logreg = joblib.load('model/logistic_regression_model.pkl')
    dt = joblib.load('model/decision_tree_model.pkl')
    rf = joblib.load('model/random_forest_model.pkl')
    nb = joblib.load('model/naive_bayes_model.pkl')
    knn = joblib.load('model/knn_model.pkl')
    xg = joblib.load('model/xgboost_model.pkl')
    sc = joblib.load('model/standard_scalar.pkl')

    model_dict = {
    'Logistic Regression': logreg,
    'Decision Tree': dt,
    'Random Forest': rf,
    'Naive Bayes': nb,
    'K-Nearest Neighbors': knn,
    'XGBoost': xg,
    'Standard scalar': sc
    }
    return model_dict[model]