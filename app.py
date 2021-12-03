from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import json
import joblib
from pandas.core.tools.numeric import to_numeric 
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing

import six
import sys
sys.modules['sklearn.externals.six'] = six

import flask

app = Flask(__name__)

main_cols = joblib.load("columns.pkl")
    

def clean_data(df_x):
    df_x['Gender'].replace({'Female':1, 'Male':0}, inplace = True)
    df_x['Geography'].replace({'France':0, 'Spain':1, 'Germany':2}, inplace = True)
    return df_x


def standardize_data(dta):
    print(dta)
    scaler = joblib.load("std_scaler.pkl")
    scaling_column = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
    dta[scaling_column] = scaler.transform(dta[scaling_column])
    print(dta)
    
    return dta


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    form_data = request.form.to_dict()
    
    df_input = pd.DataFrame.from_records([form_data], )
    df_input = df_input.drop(['submitBtn'], axis=1)
    df_input = pd.DataFrame(df_input)
    # print(df_input.dtypes)
    sample_df = pd.DataFrame(columns = main_cols)
    
    clean_df = clean_data(df_input)
    main_df = sample_df.append(clean_df)
    main_df = main_df.apply(pd.to_numeric)
    # pd.to_numeric(main_df[["CreditScore","Age","Tenure","Balance","EstimatedSalary"]],downcast='float')
    main_df = main_df.astype({"Balance":float,"EstimatedSalary":float})

    print(main_df.dtypes)
    # main_df = main_df.fillna(0)
    print(main_df)
    
         
    std_df = standardize_data(main_df)
    print(std_df)
    std_df=std_df.to_numpy()
    
    clf = joblib.load('finalized_model.pkl')
    pred = clf.predict_proba(std_df)
    print("#########")
    print(type(pred))
    print(pred)
    print(pred, pred[0], pred[0][0])
    x = round(pred[0][1]*100, 2)
    
    print(x)
    
    return flask.render_template('index.html', predicted_value="Customer Churn rate: {}%".format(str(x)))
    # return jsonify({'prediction': str(x)})

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8080)