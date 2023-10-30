import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from Main import *
import datetime

theme_plotly = None

df = pd.read_excel('data\info_res_urbanos.xlsx', sheet_name='Sheet1')


def Graphs1():

    # 1st Bar graph

    compaines_by_region = df.iloc[:, 0:2]

    fig_region = px.bar(
        compaines_by_region,
        x="Number of companies",
        # y=compaines_by_region.index,
        y="Region",
        orientation="h",
        title="<b> Total Number of Companies by Region </b>",
        color_discrete_sequence=["brown", "green", "blue"],
        color="Region",
        text_auto=True,
        template="plotly_white",
    )

    fig_region.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # 2nd line graph

    residuos_urbanos = df[['Region', 'Residuos urbanos kg por habitantes 2020', 'Residuos urbanos kg por habitantes 2019',
                           'Residuos urbanos kg por habitantes 2018', 'Residuos urbanos kg por habitantes 2017']]

    fig_residuos_urbanos = px.line(
        residuos_urbanos, x='Region', y=residuos_urbanos.columns[1:6], title='Residuos urbanos kg por habitantes')

    fig_residuos_urbanos.update_layout(

        # xaxis=dict(tickmode="linear"),
        # plot_bgcolor="rgb(0,0,0,0)",
        # yaxis=(dict(showgrid=False))
    )

    # 3rd Bar graph

    columns_percentage2020 = [
        'Region', 'Propocao de residuos em percentage 2020']
    df_percentage2020 = pd.DataFrame(df, columns=columns_percentage2020)

    # fig_percentage2020 = px.bar(df_percentage2020 , x='Region',y='Propocao de residuos em percentage 2020')

    fig_percentage2020 = px.bar(
        df_percentage2020,
        x="Propocao de residuos em percentage 2020",
        # y=compaines_by_region.index,
        y="Region",
        orientation="h",
        title="<b> Proporcao de residuos urbanos recolhidos seletivamente em percentagem 2020 </b>",
        color_discrete_sequence=["brown", "green", "blue"],
        color="Region",
        text_auto=True,
        template="plotly_white",
    )

    fig_percentage2020.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    ###### 3rd Pie chart #####

    # fig_pie = px.pie(df, values='Propocao de residuos em percentage 2020', names='Region',
    #                 title='Proporcao de residuos urbanos recolhidos seletivamente em percentagem 2020', hole=.3,
    #                 color_discrete_sequence=px.colors.sequential.RdBu)

    #fig_pie.update_layout(legend_title="Region", legend_y=0.9)

    #fig_pie.update_traces(textinfo='percent+label', textposition='inside')

    #####===== Divided in 3 columns left_column, center_column and right_column ======######

    left_column, center_column, right_column = st.columns(3)

    left_column.plotly_chart(fig_region, use_container_width=True)

    center_column.plotly_chart(fig_residuos_urbanos, use_container_width=True)

    right_column.plotly_chart(fig_percentage2020, use_container_width=True)

    # right_column.plotly_chart(
    #    fig_pie, use_container_width=True, theme=theme_plotly)

    ###### ====== Left figure for prediction of CIM Coimbra, CIM Viseu, CIM Beairs ====####

    years = [2017, 2018, 2019, 2020]
    weight_c = [427, 451, 459, 461]
    weight_v = [386, 402, 407, 427]
    weight_b = [394, 411, 413, 423]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=years, y=weight_c,
                  name='CIM Coimbra', line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=years, y=weight_v, name='CIM Viseu',
                  line=dict(color='green', width=4)))

    fig.add_trace(go.Scatter(x=years, y=weight_b, name='CIM Beiras',
                  line=dict(color='royalblue', width=4)))

    fig.update_layout(title='Residuos Urbanos kg por habitantes',
                      xaxis_title='Year', yaxis_title='Residuos Urbanos kg por habitantes')

    #####==== For prediction using Pandas Dataframe =====#####

    x = pd.DataFrame(data=years)
    y_c = pd.DataFrame(data=weight_c)
    y_v = pd.DataFrame(data=weight_v)
    y_b = pd.DataFrame(data=weight_b)

    x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(
        x, y_c, test_size=0.1)
    x_train_v, x_test_v, y_train_v, y_test_v = train_test_split(
        x, y_v, test_size=0.1)
    x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(
        x, y_b, test_size=0.1)

    reg_c = LinearRegression()
    reg_v = LinearRegression()
    reg_b = LinearRegression()

    reg_c.fit(x_train_c, y_train_c)
    reg_v.fit(x_train_v, y_train_v)
    reg_b.fit(x_train_b, y_train_b)

    reg_c.predict(x_test_c)
    reg_v.predict(x_test_v)
    reg_b.predict(x_test_b)

    x = datetime.datetime.today().year + 1  # next year taken from system
    #x = 2024

    array = np.array(x)  # input value is converted into 1 dim array

    f_value = array.astype(np.float32)

    f_value_2D = ([[f_value]])   # converted into 2D array

    my_prediction_c = reg_c.predict(f_value_2D)
    my_prediction_v = reg_v.predict(f_value_2D)
    my_prediction_b = reg_b.predict(f_value_2D)

    weight_new_c = np.array(my_prediction_c)
    weight_new_v = np.array(my_prediction_v)
    weight_new_b = np.array(my_prediction_b)

    weight_new_c = weight_new_c.item()
    weight_new_v = weight_new_v.item()
    weight_new_b = weight_new_b.item()

    years.append(x)
    #years.insert(len(years), x)

    weight_c.append(weight_new_c)
    weight_v.append(weight_new_v)
    weight_b.append(weight_new_b)

    #weight_c.insert(len(weight_c), weight_new_c)
    #weight_v.insert(len(weight_v), weight_new_v)
    #weight_b.insert(len(weight_b), weight_new_b)

    fig_pred = go.Figure()

    fig_pred.add_trace(go.Scatter(x=years, y=weight_c,
                                  name='CIM Coimbra', line=dict(color='firebrick', width=4)))

    fig_pred.add_trace(go.Scatter(x=years, y=weight_v, name='CIM Viseu',
                                  line=dict(color='green', width=4)))

    fig_pred.add_trace(go.Scatter(x=years, y=weight_b, name='CIM Beiras',
                                  line=dict(color='royalblue', width=4)))

    fig_pred.update_layout(title='The Prediction of Residuos Urbanos kg por habitantes',
                           xaxis_title='Year', yaxis_title='Residuos Urbanos kg por habitantes')

    left_column, center_column, right_column = st.columns(3)

    left_column.plotly_chart(fig, use_container_width=True)

    center_column.plotly_chart(fig_pred, use_container_width=True)

    #right_column.plotly_chart(fig_percentage2020, use_container_width=True)
