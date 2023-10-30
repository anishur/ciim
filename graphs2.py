from sqlalchemy import create_engine
import plotly.express as px
import streamlit as st
import pandas as pd
from Main import *
import pymysql
import mysql


theme_plotly = None

query = "SELECT * FROM waste_collected"

engine = create_engine(
    'mysql+pymysql://root:1234@127.0.0.1:3300/db_urban_waste').connect()

df = pd.read_sql_query(query, engine)

columns_total = ['region', 'total']
columns_papel = ['region', 'papel']
columns_plastico = ['region', 'plastico']
columns_metal = ['region', 'metal']
columns_vidro = ['region', 'vidro']
columns_madeira = ['region', 'madeira']
columns_equipamentos = ['region', 'equipamentos']
columns_pilhas = ['region', 'pilhas']
columns_oleos_alimentares = ['region', 'oleos_alimentares']
columns_outros = ['region', 'outros']

df1 = pd.DataFrame(df, columns=columns_total)
df2 = pd.DataFrame(df, columns=columns_papel)
df3 = pd.DataFrame(df, columns=columns_plastico)
df4 = pd.DataFrame(df, columns=columns_metal)
df5 = pd.DataFrame(df, columns=columns_vidro)
df6 = pd.DataFrame(df, columns=columns_madeira)
df7 = pd.DataFrame(df, columns=columns_equipamentos)
df8 = pd.DataFrame(df, columns=columns_pilhas)
df9 = pd.DataFrame(df, columns=columns_oleos_alimentares)
df10 = pd.DataFrame(df, columns=columns_outros)

#df = pd.read_excel('data\info_res_urbanos.xlsx', sheet_name='Sheet1')


def Graphs2():

    ################### 1st row graph groups start from here ###########################

    # 1st Bar graph

    #fig_1 = px.bar(df1, x='region', y='total')

    fig_1 = px.bar(
        df1,
        x="total",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Waste by Region </b>",
        color_discrete_sequence=["brown", "green", "blue"],
        color="region",
        text_auto=True,
        template="plotly_white",
    )

    fig_1.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # 2nd line graph

    residuos_urbanos = df[['region', 'total', 'papel',
                           'plastico', 'metal', 'vidro', 'madeira', 'equipamentos', 'pilhas', 'oleos_alimentares', 'outros']]

    fig_residuos_urbanos = px.line(
        residuos_urbanos, x='region', y=residuos_urbanos.columns[1:11], title='Resíduos urbanos recolhidos')

    fig_residuos_urbanos.update_layout(

        # xaxis=dict(tickmode="linear"),
        # plot_bgcolor="rgb(0,0,0,0)",
        # yaxis=(dict(showgrid=False))
    )

    # 3rd Pie Chart

    df1['percentage'] = df1['total']/df1['total'].sum()

    fig_pie = px.pie(df1, values='percentage', names='region', title='Percentage of the total generation of municipal solid waste',
                     hole=.3, color_discrete_sequence=px.colors.sequential.RdBu)

    fig_pie.update_layout(legend_title="Region", legend_y=0.9)

    fig_pie.update_traces(textinfo='percent+label', textposition='inside')

    left_column, center_column, right_column = st.columns(3)

    left_column.plotly_chart(fig_1, use_container_width=True)

    center_column.plotly_chart(fig_residuos_urbanos, use_container_width=True)

    right_column.plotly_chart(
        fig_pie, use_container_width=True, theme=theme_plotly)

    ################### 2nd row graph groups start from here ###########################

    # 2nd Bar graph

    #fig_2 = px.bar(df2, x='region',y='papel')

    fig_2 = px.bar(
        df2,
        x="papel",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Paper Waste by Region </b>",
        color_discrete_sequence=["brown", "green", "blue"],
        color="region",
        text_auto=True,
        template="plotly_white",
    )

    fig_2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # 3rd Bar graph

    #fig_3 = px.bar(df3, x='region',y='plastico')

    fig_3 = px.bar(
        df3,
        x="plastico",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Plastic Waste by Region </b>",
        color_discrete_sequence=["brown", "green", "blue"],
        color="region",
        text_auto=True,
        template="plotly_white",
    )

    fig_3.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # 4th Bar graph

    #fig_4 = px.bar(df4, x='region',y='metal')

    fig_4 = px.bar(
        df4,
        x="metal",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Metal Waste by Region </b>",
        color_discrete_sequence=["brown", "green", "blue"],
        color="region",
        text_auto=True,
        template="plotly_white",
    )

    fig_4.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    left_column, center_column, right_column = st.columns(3)

    left_column.plotly_chart(fig_2, use_container_width=True)
    center_column.plotly_chart(fig_3, use_container_width=True)
    right_column.plotly_chart(fig_4, use_container_width=True)

    ################### 3rd row graph groups start from here ###########################

    # 5th Bar graph
    #fig_5 = px.bar(df5, x='region',y='vidro')

    fig_5 = px.bar(
        df5,
        x="vidro",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Glass Waste by Region </b>",
        color_discrete_sequence=["brown", "green", "blue"],
        color="region",
        text_auto=True,
        template="plotly_white",
    )

    fig_5.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # 6th Bar graph
    #fig_6 = px.bar(df6, x='region',y='madeira')

    fig_6 = px.bar(
        df6,
        x="madeira",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Wood Waste by Region </b>",
        color_discrete_sequence=["brown", "green", "blue"],
        color="region",
        text_auto=True,
        template="plotly_white",
    )

    fig_6.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # 7th Bar graph
    # fig_7 = px.bar(df7, x='region',y='equipamentos')

    fig_7 = px.bar(
        df7,
        x="equipamentos",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Equipments Waste by Region </b>",
        color_discrete_sequence=["brown", "green", "blue"],
        color="region",
        text_auto=True,
        template="plotly_white",
    )

    fig_7.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    #####===== Divided in 3 columns left_column, center_column and right_column ======######

    left_column, center_column, right_column = st.columns(3)

    left_column.plotly_chart(fig_5, use_container_width=True)
    center_column.plotly_chart(fig_6, use_container_width=True)
    right_column.plotly_chart(fig_7, use_container_width=True)

    ################### 4th row graph groups start from here ###########################

    # 8th Bar graph
    # fig_8 = px.bar(df8, x='region',y='pilhas')

    fig_8 = px.bar(
        df8,
        x="pilhas",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Battery Waste by Region </b>",
        color_discrete_sequence=["brown", "green", "blue"],
        color="region",
        text_auto=True,
        template="plotly_white",
    )

    fig_8.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # 9th Bar graph
    # fig_9 = px.bar(df9, x='region',y='oleos_alimentares')

    fig_9 = px.bar(
        df9,
        x="oleos_alimentares",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Oleos alimentares Waste by Region </b>",
        color_discrete_sequence=["brown", "green", "blue"],
        color="region",
        text_auto=True,
        template="plotly_white",
    )

    fig_9.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # 10th Bar graph
    # fig_10 = px.bar(df10, x='region',y='outros')

    fig_10 = px.bar(
        df10,
        x="outros",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Miscellaneous Waste by Region </b>",
        color_discrete_sequence=["brown", "green", "blue"],
        color="region",
        text_auto=True,
        template="plotly_white",
    )

    fig_10.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    #####===== Divided in 3 columns left_column, center_column and right_column ======######

    left_column, center_column, right_column = st.columns(3)

    left_column.plotly_chart(fig_8, use_container_width=True)
    center_column.plotly_chart(fig_9, use_container_width=True)
    right_column.plotly_chart(fig_10, use_container_width=True)
