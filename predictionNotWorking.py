import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None

querystr = "SELECT * FROM waste_collected_by_year"

engine = create_engine(
    'mysql+pymysql://root:1234@127.0.0.1:3300/db_urban_waste_by_year').connect()

df = pd.read_sql_query(querystr, engine)


def prediction_waste():

    df1 = df.loc[df['region'] == 'CIM Coimbra']
    df2 = df.loc[df['region'] == 'CIM Viseu']
    df3 = df.loc[df['region'] == 'CIM Beiras']

    total1 = df1[['year', 'region', 'total']]
    total2 = df2[['year', 'region', 'total']]
    total3 = df3[['year', 'region', 'total']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=total1.year, y=total1.total,
                             name='CIM Coimbra', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=total2.year, y=total2.total,
                             name='CIM Viseu', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=total3.year, y=total3.total,
                             name='CIM Beiras', line=dict(color='green', width=3)))

    fig.update_layout(title='Total waste in CIM Coimbra, CIM Viseu, CIM Beiras',
                      xaxis_title='Year', yaxis_title='Total Waste (t)')

    x = pd.DataFrame(data=total1.year)

    y_c = pd.DataFrame(data=total1.total)
    y_v = pd.DataFrame(data=total2.total)
    y_b = pd.DataFrame(data=total3.total)

    predicted_year = datetime.today().year + 1

    # split data into Train and Test for the CIM Coimbra, CIM Viseu, CIM Beiras

    x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(
        x, y_c, test_size=0.1)

    x_train_v, x_test_v, y_train_v, y_test_v = train_test_split(
        x, y_v, test_size=0.1)

    x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(
        x, y_b, test_size=0.1)

    # call Linear Regression

    reg_c = LinearRegression()
    reg_v = LinearRegression()
    reg_b = LinearRegression()

    reg_c.fit(x_train_c.values, y_train_c.values)
    reg_v.fit(x_train_v.values, y_train_v.values)
    reg_b.fit(x_train_b.values, y_train_b.values)

    array = np.array(predicted_year)

    float_value = array.astype(np.float32)

    float_value_2D = ([[float_value]])

    # prediction

    prediction_c = reg_c.predict(float_value_2D)
    prediction_v = reg_v.predict(float_value_2D)
    prediction_b = reg_b.predict(float_value_2D)

    predicted_value_c = np.array(prediction_c)
    predicted_value_v = np.array(prediction_v)
    predicted_value_b = np.array(prediction_b)

    predicted_value_c = predicted_value_c.item()
    predicted_value_v = predicted_value_v.item()
    predicted_value_b = predicted_value_b.item()

    # create a new pandas dataframe with predicted values to concate with total1, total2, total3 pandas dataframe

    total1_predicted = pd.DataFrame(
        [[predicted_year, 'CIM Coimbra', predicted_value_c]])

    Total_1 = pd.concat([total1, total1_predicted], ignore_index=True)

    total2_predicted = pd.DataFrame(
        [[predicted_year, 'CIM Viseu', predicted_value_v]])

    Total_2 = pd.concat([total2, total2_predicted], ignore_index=True)

    total3_predicted = pd.DataFrame(
        [[predicted_year, 'CIM Beiras', predicted_value_b]])

    Total_3 = pd.concat([total3, total3_predicted], ignore_index=True)

    fig_total = go.Figure()

    fig_total.add_trace(go.Scatter(x=Total_1.year, y=Total_1.total,
                        name='CIM Coimbra', line=dict(color='firebrick', width=3)))
    fig_total.add_trace(go.Scatter(x=Total_2.year, y=Total_2.total,
                        name='CIM Viseu', line=dict(color='royalblue', width=3)))
    fig_total.add_trace(go.Scatter(x=Total_3.year, y=Total_3.total,
                        name='CIM Beiras', line=dict(color='green', width=3)))
    fig_total.update_layout(title='The Prediction of Total Waste in CIM Coimbra, CIM Viseu, CIM Beiras in Year ' +
                            str(predicted_year), xaxis_title='Year', yaxis_title='Total Waste (t)')

    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig, use_container_width=True)

    right_column.plotly_chart(fig_total, use_container_width=True)
