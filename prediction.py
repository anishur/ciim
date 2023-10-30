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
predicted_year = datetime.today().year + 1


def display_header():

    theme_plotly = None

    st.header(f'Waste Projections in CIM Coimbra, CIM Viseu, CIM Beiras in "{predicted_year}"')

    #display_text = f'Waste Projections in CIM Coimbra, CIM Viseu, CIM Beiras in "{predicted_year}"'

    #st.write(f'{display_text}')

    

querystr = "SELECT * FROM waste_collected_by_year"

engine = create_engine("mysql+pymysql://root:1234@127.0.0.1:3300/db_urban_waste_by_year").connect()

df =pd.read_sql_query(querystr, engine)


# predicted year to predict the value



### === The percentage increase formula function ==== ### 

def percentage(final_value, initial_value):

    res = (float(final_value) - float(initial_value)) / float(initial_value)

    return res

#### ==== prediction_total_waste funtion ===== ###

def prediction_total_waste():

    #data separated by 3-regions

    df_coimbra = df.loc[df['region'] == 'CIM Coimbra']
    df_viseu = df.loc[df['region'] == 'CIM Viseu']
    df_beiras = df.loc[df['region'] == 'CIM Beiras']

    # data collected by columns [year, region and total] in 3-regions 

    total_coimbra = df_coimbra[['year','region','total']]
    total_viseu = df_viseu[['year','region','total']]
    total_beiras = df_beiras[['year','region','total']]


    x_df = df['year'].unique()    # take one column 'year' from df and take only unique value using unique) and output comes in array
    x_df = pd.DataFrame(x_df)     # make it pandas DataFrame and output comes in rows and columns


    # take 'total' column from  df_coimbra, df_viseu, df_beiras and make it pandas DataFrame

    y_coimbra = pd.DataFrame(data=df_coimbra.total)
    y_viseu = pd.DataFrame(data=df_viseu.total)
    y_beiras = pd.DataFrame(data=df_beiras.total)

    # predicted year to predict the value

    #predicted_year = datetime.today().year + 1

    # split data into Train and Test

    x_train_coimbra, x_test_coimbra, y_train_coimbra, y_test_coimbra = train_test_split(x_df, y_coimbra, test_size = 0.1)
    x_train_viseu, x_test_viseu, y_train_viseu, y_test_viseu = train_test_split(x_df, y_viseu, test_size = 0.1)
    x_train_beiras, x_test_beiras, y_train_beiras, y_test_beiras = train_test_split(x_df, y_beiras, test_size = 0.1)

    # call Linear Regression

    reg_coimbra = LinearRegression()
    reg_viseu = LinearRegression()
    reg_beiras = LinearRegression()

    # fitting the model

    reg_coimbra.fit(x_train_coimbra, y_train_coimbra)
    reg_viseu.fit(x_train_viseu, y_train_viseu)
    reg_beiras.fit(x_train_beiras, y_train_beiras)

    array = np.array(predicted_year)
    float_value = array.astype(np.float32)
    float_value_2D = ([[float_value]])

    # Prediction

    prediction_coimbra = reg_coimbra.predict(float_value_2D)
    prediction_viseu = reg_viseu.predict(float_value_2D)
    prediction_beiras = reg_beiras.predict(float_value_2D)

    predicted_value_coimbra = np.array(prediction_coimbra)
    predicted_value_viseu = np.array(prediction_viseu)
    predicted_value_beiras = np.array(prediction_beiras)

    predicted_value_coimbra = predicted_value_coimbra.item()
    predicted_value_viseu = predicted_value_viseu.item()
    predicted_value_beiras = predicted_value_beiras.item()

    #df2 = pd.DataFrame([[2,3,4]], columns=['A','B','C']) - example of making the pandas dataframe

    total_coimbra_predicted = pd.DataFrame([[predicted_year,'CIM Coimbra', predicted_value_coimbra]], columns=['year','region','total'])
    total_viseu_predicted = pd.DataFrame([[predicted_year,'CIM Viseu', predicted_value_viseu]], columns=['year','region','total'])
    total_beiras_predicted = pd.DataFrame([[predicted_year,'CIM Beiras', predicted_value_beiras]], columns=['year','region','total'])

    #pd.concat([df1, df2], axis=0, sort=False, ignore_index=True) - example of concate two pandas dataframe

    Total_Coimbra = pd.concat([total_coimbra, total_coimbra_predicted], axis=0, sort=False, ignore_index=True) 
    Total_Viseu = pd.concat([total_viseu, total_viseu_predicted], axis=0, sort=False, ignore_index=True) 
    Total_Beiras = pd.concat([total_beiras, total_beiras_predicted], axis=0, sort=False, ignore_index=True) 

    #### === Now plot without prediction  and with prediction values in two seperate line graph to compare === ####

    # Plot line chart for 3-regions without prediction - The Total waste (t) vs. Year

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=total_coimbra.year, y=total_coimbra.total, name='CIM Coimbra', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=total_viseu.year, y=total_viseu.total, name='CIM Viseu', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=total_beiras.year, y=total_beiras.total, name='CIM Beiras', line=dict(color='green', width=3)))
    fig.update_layout(title='Total Waste in CIM Coimbra, CIM Viseu, CIM Beiras', xaxis_title = 'Year', yaxis_title = 'Total Waste (t)')


    # Plot line chart for 3-regions with prediction - The Total waste (t) vs. Year

    percentage_coimbra = percentage(Total_Coimbra.total.values[-1], total_coimbra.total.values[-1])
    percentage_viseu = percentage(Total_Viseu.total.values[-1], total_viseu.total.values[-1])
    percentage_beiras = percentage(Total_Beiras.total.values[-1], total_beiras.total.values[-1])


    fig_total = go.Figure()

    fig_total.add_trace(go.Scatter(x=Total_Coimbra.year, y=Total_Coimbra.total, name=f'CIM Coimbra: grow by "{percentage_coimbra:.1%}"', line=dict(color='firebrick', width=3)))
    fig_total.add_trace(go.Scatter(x=Total_Viseu.year, y=Total_Viseu.total, name=f'CIM Viseu: grow by "{percentage_viseu:.1%}"', line=dict(color='royalblue', width=3)))
    fig_total.add_trace(go.Scatter(x=Total_Beiras.year, y=Total_Beiras.total, name=f'CIM Beiras: grow by "{percentage_beiras:.1%}"', line=dict(color='green', width=3)))
    fig_total.update_layout(title=f'Total Waste Projections in CIM Coimbra, CIM Viseu, CIM Beiras in "{predicted_year}"', xaxis_title = 'Year', yaxis_title = 'Total Waste (t)')

    
    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig, use_container_width=True)

    right_column.plotly_chart(fig_total, use_container_width=True)


def prediction_papel_waste():

    #data separated by 3-regions

    df_coimbra = df.loc[df['region'] == 'CIM Coimbra']
    df_viseu = df.loc[df['region'] == 'CIM Viseu']
    df_beiras = df.loc[df['region'] == 'CIM Beiras']

    # data collected by columns [year, region and papel] in 3-regions 

    papel_coimbra = df_coimbra[['year','region','papel']]
    papel_viseu = df_viseu[['year','region','papel']]
    papel_beiras = df_beiras[['year','region','papel']]


    x_df = df['year'].unique()    # take one column 'year' from df and take only unique value using unique) and output comes in array
    x_df = pd.DataFrame(x_df)     # make it pandas DataFrame and output comes in rows and columns


    # take 'total' column from  df_coimbra, df_viseu, df_beiras and make it pandas DataFrame

    y_coimbra = pd.DataFrame(data=df_coimbra.papel)
    y_viseu = pd.DataFrame(data=df_viseu.papel)
    y_beiras = pd.DataFrame(data=df_beiras.papel)

    # predicted year to predict the value

    #predicted_year = datetime.today().year + 1

    # split data into Train and Test

    x_train_coimbra, x_test_coimbra, y_train_coimbra, y_test_coimbra = train_test_split(x_df, y_coimbra, test_size = 0.1)
    x_train_viseu, x_test_viseu, y_train_viseu, y_test_viseu = train_test_split(x_df, y_viseu, test_size = 0.1)
    x_train_beiras, x_test_beiras, y_train_beiras, y_test_beiras = train_test_split(x_df, y_beiras, test_size = 0.1)

    # call Linear Regression

    reg_coimbra = LinearRegression()
    reg_viseu = LinearRegression()
    reg_beiras = LinearRegression()

    # fitting the model

    reg_coimbra.fit(x_train_coimbra, y_train_coimbra)
    reg_viseu.fit(x_train_viseu, y_train_viseu)
    reg_beiras.fit(x_train_beiras, y_train_beiras)

    array = np.array(predicted_year)
    float_value = array.astype(np.float32)
    float_value_2D = ([[float_value]])

    # Prediction

    prediction_coimbra = reg_coimbra.predict(float_value_2D)
    prediction_viseu = reg_viseu.predict(float_value_2D)
    prediction_beiras = reg_beiras.predict(float_value_2D)

    predicted_value_coimbra = np.array(prediction_coimbra)
    predicted_value_viseu = np.array(prediction_viseu)
    predicted_value_beiras = np.array(prediction_beiras)

    predicted_value_coimbra = predicted_value_coimbra.item()
    predicted_value_viseu = predicted_value_viseu.item()
    predicted_value_beiras = predicted_value_beiras.item()

    #df2 = pd.DataFrame([[2,3,4]], columns=['A','B','C']) - example of making the pandas dataframe

    papel_coimbra_predicted = pd.DataFrame([[predicted_year,'CIM Coimbra', predicted_value_coimbra]], columns=['year','region','papel'])
    papel_viseu_predicted = pd.DataFrame([[predicted_year,'CIM Viseu', predicted_value_viseu]], columns=['year','region','papel'])
    papel_beiras_predicted = pd.DataFrame([[predicted_year,'CIM Beiras', predicted_value_beiras]], columns=['year','region','papel'])

    #pd.concat([df1, df2], axis=0, sort=False, ignore_index=True) - example of concate two pandas dataframe

    Papel_Coimbra = pd.concat([papel_coimbra, papel_coimbra_predicted], axis=0, sort=False, ignore_index=True) 
    Papel_Viseu = pd.concat([papel_viseu, papel_viseu_predicted], axis=0, sort=False, ignore_index=True) 
    Papel_Beiras = pd.concat([papel_beiras, papel_beiras_predicted], axis=0, sort=False, ignore_index=True) 

    #### === Now plot without prediction  and with prediction values in two seperate line graph to compare === ####

    # Plot line chart for 3-regions without prediction - The Total waste (t) vs. Year

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=papel_coimbra.year, y=papel_coimbra.papel, name='CIM Coimbra', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=papel_viseu.year, y=papel_viseu.papel, name='CIM Viseu', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=papel_beiras.year, y=papel_beiras.papel, name='CIM Beiras', line=dict(color='green', width=3)))
    fig.update_layout(title='Papel Waste in CIM Coimbra, CIM Viseu, CIM Beiras', xaxis_title = 'Year', yaxis_title = 'Papel Waste (t)')


    # Plot line chart for 3-regions with prediction - The papel waste (t) vs. Year

    percentage_coimbra = percentage(Papel_Coimbra.papel.values[-1], papel_coimbra.papel.values[-1])
    percentage_viseu = percentage(Papel_Viseu.papel.values[-1], papel_viseu.papel.values[-1])
    percentage_beiras = percentage(Papel_Beiras.papel.values[-1], papel_beiras.papel.values[-1])


    fig_total = go.Figure()

    fig_total.add_trace(go.Scatter(x=Papel_Coimbra.year, y=Papel_Coimbra.papel, name=f'CIM Coimbra: grow by "{percentage_coimbra:0.1%}"', line=dict(color='firebrick', width=3)))
    fig_total.add_trace(go.Scatter(x=Papel_Viseu.year, y=Papel_Viseu.papel, name=f'CIM Viseu: grow by "{percentage_viseu:.1%}"', line=dict(color='royalblue', width=3)))
    fig_total.add_trace(go.Scatter(x=Papel_Beiras.year, y=Papel_Beiras.papel, name=f'CIM Beiras: grow by "{percentage_beiras:.1%}"', line=dict(color='green', width=3)))
    fig_total.update_layout(title=f'Papel Waste Projections in CIM Coimbra, CIM Viseu, CIM Beiras in "{predicted_year}"', xaxis_title = 'Year', yaxis_title = 'Papel Waste (t)')

    
    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig, use_container_width=True)

    right_column.plotly_chart(fig_total, use_container_width=True)



def prediction_plastico_waste():

    #data separated by 3-regions 

    df_coimbra = df.loc[df['region'] == 'CIM Coimbra']
    df_viseu = df.loc[df['region'] == 'CIM Viseu']
    df_beiras = df.loc[df['region'] == 'CIM Beiras']

    # data collected by columns [year, region and papel] in 3-regions 

    plastico_coimbra = df_coimbra[['year','region','plastico']]
    plastico_viseu = df_viseu[['year','region','plastico']]
    plastico_beiras = df_beiras[['year','region','plastico']]


    x_df = df['year'].unique()    # take one column 'year' from df and take only unique value using unique) and output comes in array
    x_df = pd.DataFrame(x_df)     # make it pandas DataFrame and output comes in rows and columns


    # take 'total' column from  df_coimbra, df_viseu, df_beiras and make it pandas DataFrame

    y_coimbra = pd.DataFrame(data=df_coimbra.plastico)
    y_viseu = pd.DataFrame(data=df_viseu.plastico)
    y_beiras = pd.DataFrame(data=df_beiras.plastico)

    # predicted year to predict the value

    #predicted_year = datetime.today().year + 1

    # split data into Train and Test

    x_train_coimbra, x_test_coimbra, y_train_coimbra, y_test_coimbra = train_test_split(x_df, y_coimbra, test_size = 0.1)
    x_train_viseu, x_test_viseu, y_train_viseu, y_test_viseu = train_test_split(x_df, y_viseu, test_size = 0.1)
    x_train_beiras, x_test_beiras, y_train_beiras, y_test_beiras = train_test_split(x_df, y_beiras, test_size = 0.1)

    # call Linear Regression

    reg_coimbra = LinearRegression()
    reg_viseu = LinearRegression()
    reg_beiras = LinearRegression()

    # fitting the model

    reg_coimbra.fit(x_train_coimbra, y_train_coimbra)
    reg_viseu.fit(x_train_viseu, y_train_viseu)
    reg_beiras.fit(x_train_beiras, y_train_beiras)

    array = np.array(predicted_year)
    float_value = array.astype(np.float32)
    float_value_2D = ([[float_value]])

    # Prediction

    prediction_coimbra = reg_coimbra.predict(float_value_2D)
    prediction_viseu = reg_viseu.predict(float_value_2D)
    prediction_beiras = reg_beiras.predict(float_value_2D)

    predicted_value_coimbra = np.array(prediction_coimbra)
    predicted_value_viseu = np.array(prediction_viseu)
    predicted_value_beiras = np.array(prediction_beiras)

    predicted_value_coimbra = predicted_value_coimbra.item()
    predicted_value_viseu = predicted_value_viseu.item()
    predicted_value_beiras = predicted_value_beiras.item()

    #df2 = pd.DataFrame([[2,3,4]], columns=['A','B','C']) - example of making the pandas dataframe

    plastico_coimbra_predicted = pd.DataFrame([[predicted_year,'CIM Coimbra', predicted_value_coimbra]], columns=['year','region','plastico'])
    plastico_viseu_predicted = pd.DataFrame([[predicted_year,'CIM Viseu', predicted_value_viseu]], columns=['year','region','plastico'])
    plastico_beiras_predicted = pd.DataFrame([[predicted_year,'CIM Beiras', predicted_value_beiras]], columns=['year','region','plastico'])

    #pd.concat([df1, df2], axis=0, sort=False, ignore_index=True) - example of concate two pandas dataframe

    Plastico_Coimbra = pd.concat([plastico_coimbra, plastico_coimbra_predicted], axis=0, sort=False, ignore_index=True) 
    Plastico_Viseu = pd.concat([plastico_viseu, plastico_viseu_predicted], axis=0, sort=False, ignore_index=True) 
    Plastico_Beiras = pd.concat([plastico_beiras, plastico_beiras_predicted], axis=0, sort=False, ignore_index=True) 

    #### === Now plot without prediction  and with prediction values in two seperate line graph to compare === ####

    # Plot line chart for 3-regions without prediction - The Total waste (t) vs. Year

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=plastico_coimbra.year, y=plastico_coimbra.plastico, name='CIM Coimbra', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=plastico_viseu.year, y=plastico_viseu.plastico, name='CIM Viseu', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=plastico_beiras.year, y=plastico_beiras.plastico, name='CIM Beiras', line=dict(color='green', width=3)))
    fig.update_layout(title='Plastico Waste in CIM Coimbra, CIM Viseu, CIM Beiras', xaxis_title = 'Year', yaxis_title = 'Plastico Waste (t)')


    # Plot line chart for 3-regions with prediction - The plastico waste (t) vs. Year

    percentage_coimbra = percentage(Plastico_Coimbra.plastico.values[-1], plastico_coimbra.plastico.values[-1])
    percentage_viseu = percentage(Plastico_Viseu.plastico.values[-1], plastico_viseu.plastico.values[-1])
    percentage_beiras = percentage(Plastico_Beiras.plastico.values[-1], plastico_beiras.plastico.values[-1])


    fig_total = go.Figure()

    fig_total.add_trace(go.Scatter(x=Plastico_Coimbra.year, y=Plastico_Coimbra.plastico, name=f'CIM Coimbra: grow by "{percentage_coimbra:0.1%}"', line=dict(color='firebrick', width=3)))
    fig_total.add_trace(go.Scatter(x=Plastico_Viseu.year, y=Plastico_Viseu.plastico, name=f'CIM Viseu: grow by "{percentage_viseu:0.1%}"', line=dict(color='royalblue', width=3)))
    fig_total.add_trace(go.Scatter(x=Plastico_Beiras.year, y=Plastico_Beiras.plastico, name=f'CIM Beiras: grow by "{percentage_beiras:0.1%}"', line=dict(color='green', width=3)))
    fig_total.update_layout(title=f'Plastico Waste Projections in CIM Coimbra, CIM Viseu, CIM Beiras in "{predicted_year}"', xaxis_title = 'Year', yaxis_title = 'Plastico Waste (t)')

    
    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig, use_container_width=True)

    right_column.plotly_chart(fig_total, use_container_width=True)


def prediction_metal_waste():

    #data separated by 3-regions 

    df_coimbra = df.loc[df['region'] == 'CIM Coimbra']
    df_viseu = df.loc[df['region'] == 'CIM Viseu']
    df_beiras = df.loc[df['region'] == 'CIM Beiras']

    # data collected by columns [year, region and metal] in 3-regions 

    metal_coimbra = df_coimbra[['year','region','metal']]
    metal_viseu = df_viseu[['year','region','metal']]
    metal_beiras = df_beiras[['year','region','metal']]


    x_df = df['year'].unique()    # take one column 'year' from df and take only unique value using unique) and output comes in array
    x_df = pd.DataFrame(x_df)     # make it pandas DataFrame and output comes in rows and columns


    # take 'metal' column from  df_coimbra, df_viseu, df_beiras and make it pandas DataFrame

    y_coimbra = pd.DataFrame(data=df_coimbra.metal)
    y_viseu = pd.DataFrame(data=df_viseu.metal)
    y_beiras = pd.DataFrame(data=df_beiras.metal)

    # Projections year to predict the value

    #predicted_year = datetime.today().year + 1

    # split data into Train and Test

    x_train_coimbra, x_test_coimbra, y_train_coimbra, y_test_coimbra = train_test_split(x_df, y_coimbra, test_size = 0.1)
    x_train_viseu, x_test_viseu, y_train_viseu, y_test_viseu = train_test_split(x_df, y_viseu, test_size = 0.1)
    x_train_beiras, x_test_beiras, y_train_beiras, y_test_beiras = train_test_split(x_df, y_beiras, test_size = 0.1)

    # call Linear Regression

    reg_coimbra = LinearRegression()
    reg_viseu = LinearRegression()
    reg_beiras = LinearRegression()

    # fitting the model

    reg_coimbra.fit(x_train_coimbra, y_train_coimbra)
    reg_viseu.fit(x_train_viseu, y_train_viseu)
    reg_beiras.fit(x_train_beiras, y_train_beiras)

    array = np.array(predicted_year)
    float_value = array.astype(np.float32)
    float_value_2D = ([[float_value]])

    # Prediction

    prediction_coimbra = reg_coimbra.predict(float_value_2D)
    prediction_viseu = reg_viseu.predict(float_value_2D)
    prediction_beiras = reg_beiras.predict(float_value_2D)

    predicted_value_coimbra = np.array(prediction_coimbra)
    predicted_value_viseu = np.array(prediction_viseu)
    predicted_value_beiras = np.array(prediction_beiras)

    predicted_value_coimbra = predicted_value_coimbra.item()
    predicted_value_viseu = predicted_value_viseu.item()
    predicted_value_beiras = predicted_value_beiras.item()

    #df2 = pd.DataFrame([[2,3,4]], columns=['A','B','C']) - example of making the pandas dataframe

    metal_coimbra_predicted = pd.DataFrame([[predicted_year,'CIM Coimbra', predicted_value_coimbra]], columns=['year','region','metal'])
    metal_viseu_predicted = pd.DataFrame([[predicted_year,'CIM Viseu', predicted_value_viseu]], columns=['year','region','metal'])
    metal_beiras_predicted = pd.DataFrame([[predicted_year,'CIM Beiras', predicted_value_beiras]], columns=['year','region','metal'])

    #pd.concat([df1, df2], axis=0, sort=False, ignore_index=True) - example of concate two pandas dataframe

    Metal_Coimbra = pd.concat([metal_coimbra, metal_coimbra_predicted], axis=0, sort=False, ignore_index=True) 
    Metal_Viseu = pd.concat([metal_viseu, metal_viseu_predicted], axis=0, sort=False, ignore_index=True) 
    Metal_Beiras = pd.concat([metal_beiras, metal_beiras_predicted], axis=0, sort=False, ignore_index=True) 

    #### === Now plot without prediction  and with prediction values in two seperate line graph to compare === ####

    # Plot line chart for 3-regions without prediction - The metal waste (t) vs. Year

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=metal_coimbra.year, y=metal_coimbra.metal, name='CIM Coimbra', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=metal_viseu.year, y=metal_viseu.metal, name='CIM Viseu', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=metal_beiras.year, y=metal_beiras.metal, name='CIM Beiras', line=dict(color='green', width=3)))
    fig.update_layout(title='Metal Waste in CIM Coimbra, CIM Viseu, CIM Beiras', xaxis_title = 'Year', yaxis_title = 'Metal Waste (t)')


    # Plot line chart for 3-regions with prediction - The metal waste (t) vs. Year

    
    percentage_coimbra = percentage(Metal_Coimbra.metal.values[-1], metal_coimbra.metal.values[-1])
    percentage_viseu = percentage(Metal_Viseu.metal.values[-1], metal_viseu.metal.values[-1])
    percentage_beiras = percentage(Metal_Beiras.metal.values[-1], metal_beiras.metal.values[-1])


    fig_total = go.Figure()

    fig_total.add_trace(go.Scatter(x=Metal_Coimbra.year, y=Metal_Coimbra.metal, name=f'CIM Coimbra: grow by "{percentage_coimbra:0.1%}"', line=dict(color='firebrick', width=3)))
    fig_total.add_trace(go.Scatter(x=Metal_Viseu.year, y=Metal_Viseu.metal, name=f'CIM Viseu: grow by "{percentage_viseu:0.1%}"', line=dict(color='royalblue', width=3)))
    fig_total.add_trace(go.Scatter(x=Metal_Beiras.year, y=Metal_Beiras.metal, name=f'CIM Beiras: grow by "{percentage_beiras:0.1%}"', line=dict(color='green', width=3)))
    fig_total.update_layout(title=f'Metal Waste Projections in CIM Coimbra, CIM Viseu, CIM Beiras in "{predicted_year}"', xaxis_title = 'Year', yaxis_title = 'Metal Waste (t)')

    
    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig, use_container_width=True)

    right_column.plotly_chart(fig_total, use_container_width=True)



def prediction_vidro_waste():

    #data separated by 3-regions 

    df_coimbra = df.loc[df['region'] == 'CIM Coimbra']
    df_viseu = df.loc[df['region'] == 'CIM Viseu']
    df_beiras = df.loc[df['region'] == 'CIM Beiras']

    # data collected by columns [year, region and vidro] in 3-regions 

    vidro_coimbra = df_coimbra[['year','region','vidro']]
    vidro_viseu = df_viseu[['year','region','vidro']]
    vidro_beiras = df_beiras[['year','region','vidro']]


    x_df = df['year'].unique()    # take one column 'year' from df and take only unique value using unique) and output comes in array
    x_df = pd.DataFrame(x_df)     # make it pandas DataFrame and output comes in rows and columns


    # take 'vidro' column from  df_coimbra, df_viseu, df_beiras and make it pandas DataFrame

    y_coimbra = pd.DataFrame(data=df_coimbra.vidro)
    y_viseu = pd.DataFrame(data=df_viseu.vidro)
    y_beiras = pd.DataFrame(data=df_beiras.vidro)

    # predicted year to predict the value

    #predicted_year = datetime.today().year + 1

    # split data into Train and Test

    x_train_coimbra, x_test_coimbra, y_train_coimbra, y_test_coimbra = train_test_split(x_df, y_coimbra, test_size = 0.1)
    x_train_viseu, x_test_viseu, y_train_viseu, y_test_viseu = train_test_split(x_df, y_viseu, test_size = 0.1)
    x_train_beiras, x_test_beiras, y_train_beiras, y_test_beiras = train_test_split(x_df, y_beiras, test_size = 0.1)

    # call Linear Regression

    reg_coimbra = LinearRegression()
    reg_viseu = LinearRegression()
    reg_beiras = LinearRegression()

    # fitting the model

    reg_coimbra.fit(x_train_coimbra, y_train_coimbra)
    reg_viseu.fit(x_train_viseu, y_train_viseu)
    reg_beiras.fit(x_train_beiras, y_train_beiras)

    array = np.array(predicted_year)
    float_value = array.astype(np.float32)
    float_value_2D = ([[float_value]])

    # Prediction

    prediction_coimbra = reg_coimbra.predict(float_value_2D)
    prediction_viseu = reg_viseu.predict(float_value_2D)
    prediction_beiras = reg_beiras.predict(float_value_2D)

    predicted_value_coimbra = np.array(prediction_coimbra)
    predicted_value_viseu = np.array(prediction_viseu)
    predicted_value_beiras = np.array(prediction_beiras)

    predicted_value_coimbra = predicted_value_coimbra.item()
    predicted_value_viseu = predicted_value_viseu.item()
    predicted_value_beiras = predicted_value_beiras.item()

    #df2 = pd.DataFrame([[2,3,4]], columns=['A','B','C']) - example of making the pandas dataframe

    vidro_coimbra_predicted = pd.DataFrame([[predicted_year,'CIM Coimbra', predicted_value_coimbra]], columns=['year','region','vidro'])
    vidro_viseu_predicted = pd.DataFrame([[predicted_year,'CIM Viseu', predicted_value_viseu]], columns=['year','region','vidro'])
    vidro_beiras_predicted = pd.DataFrame([[predicted_year,'CIM Beiras', predicted_value_beiras]], columns=['year','region','vidro'])

    #pd.concat([df1, df2], axis=0, sort=False, ignore_index=True) - example of concate two pandas dataframe

    Vidro_Coimbra = pd.concat([vidro_coimbra, vidro_coimbra_predicted], axis=0, sort=False, ignore_index=True) 
    Vidro_Viseu = pd.concat([vidro_viseu, vidro_viseu_predicted], axis=0, sort=False, ignore_index=True) 
    Vidro_Beiras = pd.concat([vidro_beiras, vidro_beiras_predicted], axis=0, sort=False, ignore_index=True) 

    #### === Now plot without prediction  and with prediction values in two seperate line graph to compare === ####

    # Plot line chart for 3-regions without prediction - The metal waste (t) vs. Year

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=vidro_coimbra.year, y=vidro_coimbra.vidro, name='CIM Coimbra', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=vidro_viseu.year, y=vidro_viseu.vidro, name='CIM Viseu', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=vidro_beiras.year, y=vidro_beiras.vidro, name='CIM Beiras', line=dict(color='green', width=3)))
    fig.update_layout(title='Vidro Waste in CIM Coimbra, CIM Viseu, CIM Beiras', xaxis_title = 'Year', yaxis_title = 'Vidro Waste (t)')


    # Plot line chart for 3-regions with prediction - The vidro waste (t) vs. Year


    percentage_coimbra = percentage(Vidro_Coimbra.vidro.values[-1], vidro_coimbra.vidro.values[-1])
    percentage_viseu = percentage(Vidro_Viseu.vidro.values[-1], vidro_viseu.vidro.values[-1])
    percentage_beiras = percentage(Vidro_Beiras.vidro.values[-1], vidro_beiras.vidro.values[-1])


    fig_total = go.Figure()

    fig_total.add_trace(go.Scatter(x=Vidro_Coimbra.year, y=Vidro_Coimbra.vidro, name=f'CIM Coimbra: grow by "{percentage_coimbra:0.1%}"', line=dict(color='firebrick', width=3)))
    fig_total.add_trace(go.Scatter(x=Vidro_Viseu.year, y=Vidro_Viseu.vidro, name=f'CIM Viseu: grow by "{percentage_viseu:0.1%}"', line=dict(color='royalblue', width=3)))
    fig_total.add_trace(go.Scatter(x=Vidro_Beiras.year, y=Vidro_Beiras.vidro, name=f'CIM Beiras: grow by "{percentage_beiras:0.1%}"', line=dict(color='green', width=3)))
    fig_total.update_layout(title=f'Vidro Waste Projections in CIM Coimbra, CIM Viseu, CIM Beiras in "{predicted_year}"', xaxis_title = 'Year', yaxis_title = 'Vidro Waste (t)')

    
    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig, use_container_width=True)

    right_column.plotly_chart(fig_total, use_container_width=True)


def prediction_madeira_waste():

    #data separated by 3-regions 

    df_coimbra = df.loc[df['region'] == 'CIM Coimbra']
    df_viseu = df.loc[df['region'] == 'CIM Viseu']
    df_beiras = df.loc[df['region'] == 'CIM Beiras']

    # data collected by columns [year, region and madeira] in 3-regions 

    madeira_coimbra = df_coimbra[['year','region','madeira']]
    madeira_viseu = df_viseu[['year','region','madeira']]
    madeira_beiras = df_beiras[['year','region','madeira']]


    x_df = df['year'].unique()    # take one column 'year' from df and take only unique value using unique) and output comes in array
    x_df = pd.DataFrame(x_df)     # make it pandas DataFrame and output comes in rows and columns


    # take 'madeira' column from  df_coimbra, df_viseu, df_beiras and make it pandas DataFrame

    y_coimbra = pd.DataFrame(data=df_coimbra.madeira)
    y_viseu = pd.DataFrame(data=df_viseu.madeira)
    y_beiras = pd.DataFrame(data=df_beiras.madeira)

    # predicted year to predict the value

    #predicted_year = datetime.today().year + 1

    # split data into Train and Test

    x_train_coimbra, x_test_coimbra, y_train_coimbra, y_test_coimbra = train_test_split(x_df, y_coimbra, test_size = 0.1)
    x_train_viseu, x_test_viseu, y_train_viseu, y_test_viseu = train_test_split(x_df, y_viseu, test_size = 0.1)
    x_train_beiras, x_test_beiras, y_train_beiras, y_test_beiras = train_test_split(x_df, y_beiras, test_size = 0.1)

    # call Linear Regression

    reg_coimbra = LinearRegression()
    reg_viseu = LinearRegression()
    reg_beiras = LinearRegression()

    # fitting the model

    reg_coimbra.fit(x_train_coimbra, y_train_coimbra)
    reg_viseu.fit(x_train_viseu, y_train_viseu)
    reg_beiras.fit(x_train_beiras, y_train_beiras)

    array = np.array(predicted_year)
    float_value = array.astype(np.float32)
    float_value_2D = ([[float_value]])

    # Prediction

    prediction_coimbra = reg_coimbra.predict(float_value_2D)
    prediction_viseu = reg_viseu.predict(float_value_2D)
    prediction_beiras = reg_beiras.predict(float_value_2D)

    predicted_value_coimbra = np.array(prediction_coimbra)
    predicted_value_viseu = np.array(prediction_viseu)
    predicted_value_beiras = np.array(prediction_beiras)

    predicted_value_coimbra = predicted_value_coimbra.item()
    predicted_value_viseu = predicted_value_viseu.item()
    predicted_value_beiras = predicted_value_beiras.item()

    #df2 = pd.DataFrame([[2,3,4]], columns=['A','B','C']) - example of making the pandas dataframe

    madeira_coimbra_predicted = pd.DataFrame([[predicted_year,'CIM Coimbra', predicted_value_coimbra]], columns=['year','region','madeira'])
    madeira_viseu_predicted = pd.DataFrame([[predicted_year,'CIM Viseu', predicted_value_viseu]], columns=['year','region','madeira'])
    madeira_beiras_predicted = pd.DataFrame([[predicted_year,'CIM Beiras', predicted_value_beiras]], columns=['year','region','madeira'])

    #pd.concat([df1, df2], axis=0, sort=False, ignore_index=True) - example of concate two pandas dataframe

    Madeira_Coimbra = pd.concat([madeira_coimbra, madeira_coimbra_predicted], axis=0, sort=False, ignore_index=True) 
    Madeira_Viseu = pd.concat([madeira_viseu, madeira_viseu_predicted], axis=0, sort=False, ignore_index=True) 
    Madeira_Beiras = pd.concat([madeira_beiras, madeira_beiras_predicted], axis=0, sort=False, ignore_index=True) 

    #### === Now plot without prediction  and with prediction values in two seperate line graph to compare === ####

    # Plot line chart for 3-regions without prediction - The madeira waste (t) vs. Year

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=madeira_coimbra.year, y=madeira_coimbra.madeira, name='CIM Coimbra', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=madeira_viseu.year, y=madeira_viseu.madeira, name='CIM Viseu', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=madeira_beiras.year, y=madeira_beiras.madeira, name='CIM Beiras', line=dict(color='green', width=3)))
    fig.update_layout(title='Madeira Waste in CIM Coimbra, CIM Viseu, CIM Beiras', xaxis_title = 'Year', yaxis_title = 'Madeira Waste (t)')


    # Plot line chart for 3-regions with prediction - The madeira waste (t) vs. Year

    
    percentage_coimbra = percentage(Madeira_Coimbra.madeira.values[-1], madeira_coimbra.madeira.values[-1])
    percentage_viseu = percentage(Madeira_Viseu.madeira.values[-1], madeira_viseu.madeira.values[-1])
    percentage_beiras = percentage(Madeira_Beiras.madeira.values[-1], madeira_beiras.madeira.values[-1])



    fig_total = go.Figure()

    fig_total.add_trace(go.Scatter(x=Madeira_Coimbra.year, y=Madeira_Coimbra.madeira, name=f'CIM Coimbra: grow by "{percentage_coimbra:0.1%}"', line=dict(color='firebrick', width=3)))
    fig_total.add_trace(go.Scatter(x=Madeira_Viseu.year, y=Madeira_Viseu.madeira, name=f'CIM Viseu: grow by "{percentage_viseu:0.1%}"', line=dict(color='royalblue', width=3)))
    fig_total.add_trace(go.Scatter(x=Madeira_Beiras.year, y=Madeira_Beiras.madeira, name=f'CIM Beiras: grow by "{percentage_beiras:0.1%}"', line=dict(color='green', width=3)))
    fig_total.update_layout(title=f'Madeira Waste Projections in CIM Coimbra, CIM Viseu, CIM Beiras in "{predicted_year}"', xaxis_title = 'Year', yaxis_title = 'Madeira Waste (t)')

    
    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig, use_container_width=True)

    right_column.plotly_chart(fig_total, use_container_width=True)


def prediction_equipamentos_waste():

    #data separated by 3-regions 

    df_coimbra = df.loc[df['region'] == 'CIM Coimbra']
    df_viseu = df.loc[df['region'] == 'CIM Viseu']
    df_beiras = df.loc[df['region'] == 'CIM Beiras']

    # data collected by columns [year, region and equipamentos] in 3-regions 

    equipamentos_coimbra = df_coimbra[['year','region','equipamentos']]
    equipamentos_viseu = df_viseu[['year','region','equipamentos']]
    equipamentos_beiras = df_beiras[['year','region','equipamentos']]


    x_df = df['year'].unique()    # take one column 'year' from df and take only unique value using unique) and output comes in array
    x_df = pd.DataFrame(x_df)     # make it pandas DataFrame and output comes in rows and columns

    # take 'equipamentos' column from  df_coimbra, df_viseu, df_beiras and make it pandas DataFrame

    y_coimbra = pd.DataFrame(data=df_coimbra.equipamentos)
    y_viseu = pd.DataFrame(data=df_viseu.equipamentos)
    y_beiras = pd.DataFrame(data=df_beiras.equipamentos)

    # predicted year to predict the value

    #predicted_year = datetime.today().year + 1

    # split data into Train and Test

    x_train_coimbra, x_test_coimbra, y_train_coimbra, y_test_coimbra = train_test_split(x_df, y_coimbra, test_size = 0.1)
    x_train_viseu, x_test_viseu, y_train_viseu, y_test_viseu = train_test_split(x_df, y_viseu, test_size = 0.1)
    x_train_beiras, x_test_beiras, y_train_beiras, y_test_beiras = train_test_split(x_df, y_beiras, test_size = 0.1)

    # call Linear Regression

    reg_coimbra = LinearRegression()
    reg_viseu = LinearRegression()
    reg_beiras = LinearRegression()

    # fitting the model

    reg_coimbra.fit(x_train_coimbra, y_train_coimbra)
    reg_viseu.fit(x_train_viseu, y_train_viseu)
    reg_beiras.fit(x_train_beiras, y_train_beiras)

    array = np.array(predicted_year)
    float_value = array.astype(np.float32)
    float_value_2D = ([[float_value]])

    # Prediction

    prediction_coimbra = reg_coimbra.predict(float_value_2D)
    prediction_viseu = reg_viseu.predict(float_value_2D)
    prediction_beiras = reg_beiras.predict(float_value_2D)

    predicted_value_coimbra = np.array(prediction_coimbra)
    predicted_value_viseu = np.array(prediction_viseu)
    predicted_value_beiras = np.array(prediction_beiras)

    predicted_value_coimbra = predicted_value_coimbra.item()
    predicted_value_viseu = predicted_value_viseu.item()
    predicted_value_beiras = predicted_value_beiras.item()

    #df2 = pd.DataFrame([[2,3,4]], columns=['A','B','C']) - example of making the pandas dataframe

    equipamentos_coimbra_predicted = pd.DataFrame([[predicted_year,'CIM Coimbra', predicted_value_coimbra]], columns=['year','region','equipamentos'])
    equipamentos_viseu_predicted = pd.DataFrame([[predicted_year,'CIM Viseu', predicted_value_viseu]], columns=['year','region','equipamentos'])
    equipamentos_beiras_predicted = pd.DataFrame([[predicted_year,'CIM Beiras', predicted_value_beiras]], columns=['year','region','equipamentos'])

    #pd.concat([df1, df2], axis=0, sort=False, ignore_index=True) - example of concate two pandas dataframe

    Equipamentos_Coimbra = pd.concat([equipamentos_coimbra, equipamentos_coimbra_predicted], axis=0, sort=False, ignore_index=True) 
    Equipamentos_Viseu = pd.concat([equipamentos_viseu, equipamentos_viseu_predicted], axis=0, sort=False, ignore_index=True) 
    Equipamentos_Beiras = pd.concat([equipamentos_beiras, equipamentos_beiras_predicted], axis=0, sort=False, ignore_index=True) 

    #### === Now plot without prediction  and with prediction values in two seperate line graph to compare === ####

    # Plot line chart for 3-regions without prediction - The madeira waste (t) vs. Year

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=equipamentos_coimbra.year, y=equipamentos_coimbra.equipamentos, name='CIM Coimbra', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=equipamentos_viseu.year, y=equipamentos_viseu.equipamentos, name='CIM Viseu', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=equipamentos_beiras.year, y=equipamentos_beiras.equipamentos, name='CIM Beiras', line=dict(color='green', width=3)))
    fig.update_layout(title='Equipamentos Waste in CIM Coimbra, CIM Viseu, CIM Beiras', xaxis_title = 'Year', yaxis_title = 'Equipamentos Waste (t)')


    # Plot line chart for 3-regions with prediction - The madeira waste (t) vs. Year

        
    percentage_coimbra = percentage(Equipamentos_Coimbra.equipamentos.values[-1], equipamentos_coimbra.equipamentos.values[-1])
    percentage_viseu = percentage(Equipamentos_Viseu.equipamentos.values[-1], equipamentos_viseu.equipamentos.values[-1])
    percentage_beiras = percentage(Equipamentos_Beiras.equipamentos.values[-1], equipamentos_beiras.equipamentos.values[-1])


    fig_total = go.Figure()

    fig_total.add_trace(go.Scatter(x=Equipamentos_Coimbra.year, y=Equipamentos_Coimbra.equipamentos, name=f'CIM Coimbra: grow by "{percentage_coimbra:0.1%}"', line=dict(color='firebrick', width=3)))
    fig_total.add_trace(go.Scatter(x=Equipamentos_Viseu.year, y=Equipamentos_Viseu.equipamentos, name=f'CIM Viseu: grow by "{percentage_viseu:0.1%}"', line=dict(color='royalblue', width=3)))
    fig_total.add_trace(go.Scatter(x=Equipamentos_Beiras.year, y=Equipamentos_Beiras.equipamentos, name=f'CIM Beiras: grow by "{percentage_beiras:0.1%}"', line=dict(color='green', width=3)))
    fig_total.update_layout(title=f'Equipamentos Waste Projections in CIM Coimbra, CIM Viseu, CIM Beiras in "{predicted_year}"', xaxis_title = 'Year', yaxis_title = 'Equipamentos Waste (t)')

    
    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig, use_container_width=True)

    right_column.plotly_chart(fig_total, use_container_width=True)



def prediction_pilhas_waste():

    #data separated by 3-regions 

    df_coimbra = df.loc[df['region'] == 'CIM Coimbra']
    df_viseu = df.loc[df['region'] == 'CIM Viseu']
    df_beiras = df.loc[df['region'] == 'CIM Beiras']

    # data collected by columns [year, region and pilhas] in 3-regions 

    pilhas_coimbra = df_coimbra[['year','region','pilhas']]
    pilhas_viseu = df_viseu[['year','region','pilhas']]
    pilhas_beiras = df_beiras[['year','region','pilhas']]

    x_df = df['year'].unique()    # take one column 'year' from df and take only unique value using unique) and output comes in array
    x_df = pd.DataFrame(x_df)     # make it pandas DataFrame and output comes in rows and columns

    # take 'pilhas' column from  df_coimbra, df_viseu, df_beiras and make it pandas DataFrame

    y_coimbra = pd.DataFrame(data=df_coimbra.pilhas)
    y_viseu = pd.DataFrame(data=df_viseu.pilhas)
    y_beiras = pd.DataFrame(data=df_beiras.pilhas)

    # predicted year to predict the value

    #predicted_year = datetime.today().year + 1

    # split data into Train and Test

    x_train_coimbra, x_test_coimbra, y_train_coimbra, y_test_coimbra = train_test_split(x_df, y_coimbra, test_size = 0.1)
    x_train_viseu, x_test_viseu, y_train_viseu, y_test_viseu = train_test_split(x_df, y_viseu, test_size = 0.1)
    x_train_beiras, x_test_beiras, y_train_beiras, y_test_beiras = train_test_split(x_df, y_beiras, test_size = 0.1)

    # call Linear Regression

    reg_coimbra = LinearRegression()
    reg_viseu = LinearRegression()
    reg_beiras = LinearRegression()

    # fitting the model

    reg_coimbra.fit(x_train_coimbra, y_train_coimbra)
    reg_viseu.fit(x_train_viseu, y_train_viseu)
    reg_beiras.fit(x_train_beiras, y_train_beiras)

    array = np.array(predicted_year)
    float_value = array.astype(np.float32)
    float_value_2D = ([[float_value]])

    # Prediction

    prediction_coimbra = reg_coimbra.predict(float_value_2D)
    prediction_viseu = reg_viseu.predict(float_value_2D)
    prediction_beiras = reg_beiras.predict(float_value_2D)

    predicted_value_coimbra = np.array(prediction_coimbra)
    predicted_value_viseu = np.array(prediction_viseu)
    predicted_value_beiras = np.array(prediction_beiras)

    predicted_value_coimbra = predicted_value_coimbra.item()
    predicted_value_viseu = predicted_value_viseu.item()
    predicted_value_beiras = predicted_value_beiras.item()

    #df2 = pd.DataFrame([[2,3,4]], columns=['A','B','C']) - example of making the pandas dataframe

    pilhas_coimbra_predicted = pd.DataFrame([[predicted_year,'CIM Coimbra', predicted_value_coimbra]], columns=['year','region','pilhas'])
    pilhas_viseu_predicted = pd.DataFrame([[predicted_year,'CIM Viseu', predicted_value_viseu]], columns=['year','region','pilhas'])
    pilhas_beiras_predicted = pd.DataFrame([[predicted_year,'CIM Beiras', predicted_value_beiras]], columns=['year','region','pilhas'])

    #pd.concat([df1, df2], axis=0, sort=False, ignore_index=True) - example of concate two pandas dataframe

    Pilhas_Coimbra = pd.concat([pilhas_coimbra, pilhas_coimbra_predicted], axis=0, sort=False, ignore_index=True) 
    Pilhas_Viseu = pd.concat([pilhas_viseu, pilhas_viseu_predicted], axis=0, sort=False, ignore_index=True) 
    Pilhas_Beiras = pd.concat([pilhas_beiras, pilhas_beiras_predicted], axis=0, sort=False, ignore_index=True) 

    #### === Now plot without prediction  and with prediction values in two seperate line graph to compare === ####

    # Plot line chart for 3-regions without prediction - The pilhas waste (t) vs. Year

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=pilhas_coimbra.year, y=pilhas_coimbra.pilhas, name='CIM Coimbra', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=pilhas_viseu.year, y=pilhas_viseu.pilhas, name='CIM Viseu', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=pilhas_beiras.year, y=pilhas_beiras.pilhas, name='CIM Beiras', line=dict(color='green', width=3)))
    fig.update_layout(title='Pilhas Waste in CIM Coimbra, CIM Viseu, CIM Beiras', xaxis_title = 'Year', yaxis_title = 'Pilhas Waste (t)')

    # Plot line chart for 3-regions with prediction - The pilhas waste (t) vs. Year

        
    percentage_coimbra = percentage(Pilhas_Coimbra.pilhas.values[-1], pilhas_coimbra.pilhas.values[-1])
    percentage_viseu = percentage(Pilhas_Viseu.pilhas.values[-1], pilhas_viseu.pilhas.values[-1])
    percentage_beiras = percentage(Pilhas_Beiras.pilhas.values[-1], pilhas_beiras.pilhas.values[-1])


    fig_total = go.Figure()

    fig_total.add_trace(go.Scatter(x=Pilhas_Coimbra.year, y=Pilhas_Coimbra.pilhas, name=f'CIM Coimbra: grow by "{percentage_coimbra:0.1%}"', line=dict(color='firebrick', width=3)))
    fig_total.add_trace(go.Scatter(x=Pilhas_Viseu.year, y=Pilhas_Viseu.pilhas, name=f'CIM Viseu: grow by "{percentage_viseu:0.1%}"', line=dict(color='royalblue', width=3)))
    fig_total.add_trace(go.Scatter(x=Pilhas_Beiras.year, y=Pilhas_Beiras.pilhas, name=f'CIM Beiras: grow by "{percentage_beiras:0.1%}"', line=dict(color='green', width=3)))
    fig_total.update_layout(title=f'Pilhas Waste Projections in CIM Coimbra, CIM Viseu, CIM Beiras in "{predicted_year}"', xaxis_title = 'Year', yaxis_title = 'Pilhas Waste (t)')

    
    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig, use_container_width=True)

    right_column.plotly_chart(fig_total, use_container_width=True)



def prediction_oleos_alimentares_waste():

    #data separated by 3-regions 

    df_coimbra = df.loc[df['region'] == 'CIM Coimbra']
    df_viseu = df.loc[df['region'] == 'CIM Viseu']
    df_beiras = df.loc[df['region'] == 'CIM Beiras']

    # data collected by columns [year, region and oleos_alimentares] in 3-regions 

    oleos_alimentares_coimbra = df_coimbra[['year','region','oleos_alimentares']]
    oleos_alimentares_viseu = df_viseu[['year','region','oleos_alimentares']]
    oleos_alimentares_beiras = df_beiras[['year','region','oleos_alimentares']]

    x_df = df['year'].unique()    # take one column 'year' from df and take only unique value using unique) and output comes in array
    x_df = pd.DataFrame(x_df)     # make it pandas DataFrame and output comes in rows and columns

    # take 'oleos_alimentares' column from  df_coimbra, df_viseu, df_beiras and make it pandas DataFrame

    y_coimbra = pd.DataFrame(data=df_coimbra.oleos_alimentares)
    y_viseu = pd.DataFrame(data=df_viseu.oleos_alimentares)
    y_beiras = pd.DataFrame(data=df_beiras.oleos_alimentares)

    # predicted year to predict the value

    #predicted_year = datetime.today().year + 1

    # split data into Train and Test

    x_train_coimbra, x_test_coimbra, y_train_coimbra, y_test_coimbra = train_test_split(x_df, y_coimbra, test_size = 0.1)
    x_train_viseu, x_test_viseu, y_train_viseu, y_test_viseu = train_test_split(x_df, y_viseu, test_size = 0.1)
    x_train_beiras, x_test_beiras, y_train_beiras, y_test_beiras = train_test_split(x_df, y_beiras, test_size = 0.1)

    # call Linear Regression

    reg_coimbra = LinearRegression()
    reg_viseu = LinearRegression()
    reg_beiras = LinearRegression()

    # fitting the model

    reg_coimbra.fit(x_train_coimbra, y_train_coimbra)
    reg_viseu.fit(x_train_viseu, y_train_viseu)
    reg_beiras.fit(x_train_beiras, y_train_beiras)

    array = np.array(predicted_year)
    float_value = array.astype(np.float32)
    float_value_2D = ([[float_value]])

    # Prediction

    prediction_coimbra = reg_coimbra.predict(float_value_2D)
    prediction_viseu = reg_viseu.predict(float_value_2D)
    prediction_beiras = reg_beiras.predict(float_value_2D)

    predicted_value_coimbra = np.array(prediction_coimbra)
    predicted_value_viseu = np.array(prediction_viseu)
    predicted_value_beiras = np.array(prediction_beiras)

    predicted_value_coimbra = predicted_value_coimbra.item()
    predicted_value_viseu = predicted_value_viseu.item()
    predicted_value_beiras = predicted_value_beiras.item()

    #df2 = pd.DataFrame([[2,3,4]], columns=['A','B','C']) - example of making the pandas dataframe

    oleos_alimentares_coimbra_predicted = pd.DataFrame([[predicted_year,'CIM Coimbra', predicted_value_coimbra]], columns=['year','region','oleos_alimentares'])
    oleos_alimentares_viseu_predicted = pd.DataFrame([[predicted_year,'CIM Viseu', predicted_value_viseu]], columns=['year','region','oleos_alimentares'])
    oleos_alimentares_beiras_predicted = pd.DataFrame([[predicted_year,'CIM Beiras', predicted_value_beiras]], columns=['year','region','oleos_alimentares'])

    #pd.concat([df1, df2], axis=0, sort=False, ignore_index=True) - example of concate two pandas dataframe

    Oleos_alimentares_Coimbra = pd.concat([oleos_alimentares_coimbra, oleos_alimentares_coimbra_predicted], axis=0, sort=False, ignore_index=True) 
    Oleos_alimentares_Viseu = pd.concat([oleos_alimentares_viseu, oleos_alimentares_viseu_predicted], axis=0, sort=False, ignore_index=True) 
    Oleos_alimentares_Beiras = pd.concat([oleos_alimentares_beiras, oleos_alimentares_beiras_predicted], axis=0, sort=False, ignore_index=True) 

    #### === Now plot without prediction  and with prediction values in two seperate line graph to compare === ####

    # Plot line chart for 3-regions without prediction - The oleos_alimentares waste (t) vs. Year

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=oleos_alimentares_coimbra.year, y=oleos_alimentares_coimbra.oleos_alimentares, name='CIM Coimbra', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=oleos_alimentares_viseu.year, y=oleos_alimentares_viseu.oleos_alimentares, name='CIM Viseu', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=oleos_alimentares_beiras.year, y=oleos_alimentares_beiras.oleos_alimentares, name='CIM Beiras', line=dict(color='green', width=3)))
    fig.update_layout(title='Oleos_alimentares Waste in CIM Coimbra, CIM Viseu, CIM Beiras', xaxis_title = 'Year', yaxis_title = 'Oleos_alimentares Waste (t)')

    # Plot line chart for 3-regions with prediction - The Oleos_alimentares waste (t) vs. Year

            
    percentage_coimbra = percentage(Oleos_alimentares_Coimbra.oleos_alimentares.values[-1], oleos_alimentares_coimbra.oleos_alimentares.values[-1])
    percentage_viseu = percentage(Oleos_alimentares_Viseu.oleos_alimentares.values[-1], oleos_alimentares_viseu.oleos_alimentares.values[-1])
    percentage_beiras = percentage(Oleos_alimentares_Beiras.oleos_alimentares.values[-1], oleos_alimentares_beiras.oleos_alimentares.values[-1])


    fig_total = go.Figure()

    fig_total.add_trace(go.Scatter(x=Oleos_alimentares_Coimbra.year, y=Oleos_alimentares_Coimbra.oleos_alimentares, name=f'CIM Coimbra: grow by "{percentage_coimbra:0.1%}"', line=dict(color='firebrick', width=3)))
    fig_total.add_trace(go.Scatter(x=Oleos_alimentares_Viseu.year, y=Oleos_alimentares_Viseu.oleos_alimentares, name=f'CIM Viseu: grow by "{percentage_viseu:0.1%}"', line=dict(color='royalblue', width=3)))
    fig_total.add_trace(go.Scatter(x=Oleos_alimentares_Beiras.year, y=Oleos_alimentares_Beiras.oleos_alimentares, name=f'CIM Beiras: grow by "{percentage_beiras:0.1%}"', line=dict(color='green', width=3)))
    fig_total.update_layout(title=f'Oleos_alimentares Waste Projections in CIM Coimbra, CIM Viseu, CIM Beiras in "{predicted_year}"', xaxis_title = 'Year', yaxis_title = 'Oleos_alimentares Waste (t)')

    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig, use_container_width=True)

    right_column.plotly_chart(fig_total, use_container_width=True)


def prediction_outros_waste():

    #data separated by 3-regions 

    df_coimbra = df.loc[df['region'] == 'CIM Coimbra']
    df_viseu = df.loc[df['region'] == 'CIM Viseu']
    df_beiras = df.loc[df['region'] == 'CIM Beiras']

    # data collected by columns [year, region and outros] in 3-regions 

    outros_coimbra = df_coimbra[['year','region','outros']]
    outros_viseu = df_viseu[['year','region','outros']]
    outros_beiras = df_beiras[['year','region','outros']]

    x_df = df['year'].unique()    # take one column 'year' from df and take only unique value using unique) and output comes in array
    x_df = pd.DataFrame(x_df)     # make it pandas DataFrame and output comes in rows and columns

    # take 'outros' column from  df_coimbra, df_viseu, df_beiras and make it pandas DataFrame

    y_coimbra = pd.DataFrame(data=df_coimbra.outros)
    y_viseu = pd.DataFrame(data=df_viseu.outros)
    y_beiras = pd.DataFrame(data=df_beiras.outros)

    # predicted year to predict the value

    #predicted_year = datetime.today().year + 1

    # split data into Train and Test

    x_train_coimbra, x_test_coimbra, y_train_coimbra, y_test_coimbra = train_test_split(x_df, y_coimbra, test_size = 0.1)
    x_train_viseu, x_test_viseu, y_train_viseu, y_test_viseu = train_test_split(x_df, y_viseu, test_size = 0.1)
    x_train_beiras, x_test_beiras, y_train_beiras, y_test_beiras = train_test_split(x_df, y_beiras, test_size = 0.1)

    # call Linear Regression

    reg_coimbra = LinearRegression()
    reg_viseu = LinearRegression()
    reg_beiras = LinearRegression()

    # fitting the model

    reg_coimbra.fit(x_train_coimbra, y_train_coimbra)
    reg_viseu.fit(x_train_viseu, y_train_viseu)
    reg_beiras.fit(x_train_beiras, y_train_beiras)

    array = np.array(predicted_year)
    float_value = array.astype(np.float32)
    float_value_2D = ([[float_value]])

    # Prediction

    prediction_coimbra = reg_coimbra.predict(float_value_2D)
    prediction_viseu = reg_viseu.predict(float_value_2D)
    prediction_beiras = reg_beiras.predict(float_value_2D)

    predicted_value_coimbra = np.array(prediction_coimbra)
    predicted_value_viseu = np.array(prediction_viseu)
    predicted_value_beiras = np.array(prediction_beiras)

    predicted_value_coimbra = predicted_value_coimbra.item()
    predicted_value_viseu = predicted_value_viseu.item()
    predicted_value_beiras = predicted_value_beiras.item()

    #df2 = pd.DataFrame([[2,3,4]], columns=['A','B','C']) - example of making the pandas dataframe

    outros_coimbra_predicted = pd.DataFrame([[predicted_year,'CIM Coimbra', predicted_value_coimbra]], columns=['year','region','outros'])
    outros_viseu_predicted = pd.DataFrame([[predicted_year,'CIM Viseu', predicted_value_viseu]], columns=['year','region','outros'])
    outros_beiras_predicted = pd.DataFrame([[predicted_year,'CIM Beiras', predicted_value_beiras]], columns=['year','region','outros'])

    #pd.concat([df1, df2], axis=0, sort=False, ignore_index=True) - example of concate two pandas dataframe

    Outros_Coimbra = pd.concat([outros_coimbra, outros_coimbra_predicted], axis=0, sort=False, ignore_index=True) 
    Outros_Viseu = pd.concat([outros_viseu, outros_viseu_predicted], axis=0, sort=False, ignore_index=True) 
    Outros_Beiras = pd.concat([outros_beiras, outros_beiras_predicted], axis=0, sort=False, ignore_index=True) 

    #### === Now plot without prediction  and with prediction values in two seperate line graph to compare === ####

    # Plot line chart for 3-regions without prediction - The oleos_alimentares waste (t) vs. Year

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=outros_coimbra.year, y=outros_coimbra.outros, name='CIM Coimbra', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=outros_viseu.year, y=outros_viseu.outros, name='CIM Viseu', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=outros_beiras.year, y=outros_beiras.outros, name='CIM Beiras', line=dict(color='green', width=3)))
    fig.update_layout(title='Outros Waste in CIM Coimbra, CIM Viseu, CIM Beiras', xaxis_title = 'Year', yaxis_title = 'Outros Waste (t)')

    # Plot line chart for 3-regions with prediction - The Outros waste (t) vs. Year

    percentage_coimbra = percentage(Outros_Coimbra.outros.values[-1], outros_coimbra.outros.values[-1])
    percentage_viseu = percentage(Outros_Viseu.outros.values[-1], outros_viseu.outros.values[-1])
    percentage_beiras = percentage(Outros_Beiras.outros.values[-1], outros_beiras.outros.values[-1])

    fig_total = go.Figure()

    fig_total.add_trace(go.Scatter(x=Outros_Coimbra.year, y=Outros_Coimbra.outros, name=f'CIM Coimbra: grow by "{percentage_coimbra:0.1%}"', line=dict(color='firebrick', width=3)))
    fig_total.add_trace(go.Scatter(x=Outros_Viseu.year, y=Outros_Viseu.outros, name=f'CIM Viseu: grow by "{percentage_viseu:0.1%}"', line=dict(color='royalblue', width=3)))
    fig_total.add_trace(go.Scatter(x=Outros_Beiras.year, y=Outros_Beiras.outros, name=f'CIM Beiras: grow by "{percentage_beiras:0.1%}"', line=dict(color='green', width=3)))
    fig_total.update_layout(title=f'Outros Waste Projections in CIM Coimbra, CIM Viseu, CIM Beiras in "{predicted_year}"', xaxis_title = 'Year', yaxis_title = 'Outros Waste (t)')

    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig, use_container_width=True)

    right_column.plotly_chart(fig_total, use_container_width=True)


def prediction_recolha_indiferenciada_waste():

    #data separated by 3-regions 

    df_coimbra = df.loc[df['region'] == 'CIM Coimbra']
    df_viseu = df.loc[df['region'] == 'CIM Viseu']
    df_beiras = df.loc[df['region'] == 'CIM Beiras']

    # data collected by columns [year, region and recolha_indiferenciada] in 3-regions 

    recolha_indiferenciada_coimbra = df_coimbra[['year','region','recolha_indiferenciada']]
    recolha_indiferenciada_viseu = df_viseu[['year','region','recolha_indiferenciada']]
    recolha_indiferenciada_beiras = df_beiras[['year','region','recolha_indiferenciada']]

    x_df = df['year'].unique()    # take one column 'year' from df and take only unique value using unique) and output comes in array
    x_df = pd.DataFrame(x_df)     # make it pandas DataFrame and output comes in rows and columns

    # take 'outros' column from  df_coimbra, df_viseu, df_beiras and make it pandas DataFrame

    y_coimbra = pd.DataFrame(data=df_coimbra.recolha_indiferenciada)
    y_viseu = pd.DataFrame(data=df_viseu.recolha_indiferenciada)
    y_beiras = pd.DataFrame(data=df_beiras.recolha_indiferenciada)

    # predicted year to predict the value

    #predicted_year = datetime.today().year + 1

    # split data into Train and Test

    x_train_coimbra, x_test_coimbra, y_train_coimbra, y_test_coimbra = train_test_split(x_df, y_coimbra, test_size = 0.1)
    x_train_viseu, x_test_viseu, y_train_viseu, y_test_viseu = train_test_split(x_df, y_viseu, test_size = 0.1)
    x_train_beiras, x_test_beiras, y_train_beiras, y_test_beiras = train_test_split(x_df, y_beiras, test_size = 0.1)

    # call Linear Regression

    reg_coimbra = LinearRegression()
    reg_viseu = LinearRegression()
    reg_beiras = LinearRegression()

    # fitting the model

    reg_coimbra.fit(x_train_coimbra, y_train_coimbra)
    reg_viseu.fit(x_train_viseu, y_train_viseu)
    reg_beiras.fit(x_train_beiras, y_train_beiras)

    array = np.array(predicted_year)
    float_value = array.astype(np.float32)
    float_value_2D = ([[float_value]])

    # Prediction

    prediction_coimbra = reg_coimbra.predict(float_value_2D)
    prediction_viseu = reg_viseu.predict(float_value_2D)
    prediction_beiras = reg_beiras.predict(float_value_2D)

    predicted_value_coimbra = np.array(prediction_coimbra)
    predicted_value_viseu = np.array(prediction_viseu)
    predicted_value_beiras = np.array(prediction_beiras)

    predicted_value_coimbra = predicted_value_coimbra.item()
    predicted_value_viseu = predicted_value_viseu.item()
    predicted_value_beiras = predicted_value_beiras.item()

    #df2 = pd.DataFrame([[2,3,4]], columns=['A','B','C']) - example of making the pandas dataframe

    recolha_indiferenciada_coimbra_predicted = pd.DataFrame([[predicted_year,'CIM Coimbra', predicted_value_coimbra]], columns=['year','region','recolha_indiferenciada'])
    recolha_indiferenciada_viseu_predicted = pd.DataFrame([[predicted_year,'CIM Viseu', predicted_value_viseu]], columns=['year','region','recolha_indiferenciada'])
    recolha_indiferenciada_beiras_predicted = pd.DataFrame([[predicted_year,'CIM Beiras', predicted_value_beiras]], columns=['year','region','recolha_indiferenciada'])

    #pd.concat([df1, df2], axis=0, sort=False, ignore_index=True) - example of concate two pandas dataframe

    Recolha_indiferenciada_Coimbra = pd.concat([recolha_indiferenciada_coimbra, recolha_indiferenciada_coimbra_predicted], axis=0, sort=False, ignore_index=True) 
    Recolha_indiferenciada_Viseu = pd.concat([recolha_indiferenciada_viseu, recolha_indiferenciada_viseu_predicted], axis=0, sort=False, ignore_index=True) 
    Recolha_indiferenciada_Beiras = pd.concat([recolha_indiferenciada_beiras, recolha_indiferenciada_beiras_predicted], axis=0, sort=False, ignore_index=True) 

    #### === Now plot without prediction  and with prediction values in two seperate line graph to compare === ####

    # Plot line chart for 3-regions without prediction - The recolha_indiferenciada waste (t) vs. Year

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=recolha_indiferenciada_coimbra.year, y=recolha_indiferenciada_coimbra.recolha_indiferenciada, name='CIM Coimbra', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=recolha_indiferenciada_viseu.year, y=recolha_indiferenciada_viseu.recolha_indiferenciada, name='CIM Viseu', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=recolha_indiferenciada_beiras.year, y=recolha_indiferenciada_beiras.recolha_indiferenciada, name='CIM Beiras', line=dict(color='green', width=3)))
    fig.update_layout(title='Recolha_indiferenciada Waste in CIM Coimbra, CIM Viseu, CIM Beiras', xaxis_title = 'Year', yaxis_title = 'Recolha_indiferenciada Waste (t)')

    # Plot line chart for 3-regions with prediction - The Recolha_indiferenciada waste (t) vs. Year


    percentage_coimbra = percentage(Recolha_indiferenciada_Coimbra.recolha_indiferenciada.values[-1], recolha_indiferenciada_coimbra.recolha_indiferenciada.values[-1])
    percentage_viseu = percentage(Recolha_indiferenciada_Viseu.recolha_indiferenciada.values[-1], recolha_indiferenciada_viseu.recolha_indiferenciada.values[-1])
    percentage_beiras = percentage(Recolha_indiferenciada_Beiras.recolha_indiferenciada.values[-1], recolha_indiferenciada_beiras.recolha_indiferenciada.values[-1])


    fig_total = go.Figure()

    fig_total.add_trace(go.Scatter(x=Recolha_indiferenciada_Coimbra.year, y=Recolha_indiferenciada_Coimbra.recolha_indiferenciada, name=f'CIM Coimbra: grow by "{percentage_coimbra:0.1%}"', line=dict(color='firebrick', width=3)))
    fig_total.add_trace(go.Scatter(x=Recolha_indiferenciada_Viseu.year, y=Recolha_indiferenciada_Viseu.recolha_indiferenciada, name=f'CIM Viseu: grow by "{percentage_viseu:0.1%}"', line=dict(color='royalblue', width=3)))
    fig_total.add_trace(go.Scatter(x=Recolha_indiferenciada_Beiras.year, y=Recolha_indiferenciada_Beiras.recolha_indiferenciada, name=f'CIM Beiras: grow by "{percentage_beiras:0.1%}"', line=dict(color='green', width=3)))
    fig_total.update_layout(title=f'Recolha_indiferenciada Waste Projections in CIM Coimbra, CIM Viseu, CIM Beiras in "{predicted_year}"', xaxis_title = 'Year', yaxis_title = 'Recolha_indiferenciada Waste (t)')

    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig, use_container_width=True)

    right_column.plotly_chart(fig_total, use_container_width=True)



def prediction_recolha_selectiva_waste():

    #data separated by 3-regions 

    df_coimbra = df.loc[df['region'] == 'CIM Coimbra']
    df_viseu = df.loc[df['region'] == 'CIM Viseu']
    df_beiras = df.loc[df['region'] == 'CIM Beiras']

    # data collected by columns [year, region and recolha_selectiva] in 3-regions 

    recolha_selectiva_coimbra = df_coimbra[['year','region','recolha_selectiva']]
    recolha_selectiva_viseu = df_viseu[['year','region','recolha_selectiva']]
    recolha_selectiva_beiras = df_beiras[['year','region','recolha_selectiva']]

    x_df = df['year'].unique()    # take one column 'year' from df and take only unique value using unique) and output comes in array
    x_df = pd.DataFrame(x_df)     # make it pandas DataFrame and output comes in rows and columns

    # take 'selectiva' column from  df_coimbra, df_viseu, df_beiras and make it pandas DataFrame

    y_coimbra = pd.DataFrame(data=df_coimbra.recolha_selectiva)
    y_viseu = pd.DataFrame(data=df_viseu.recolha_selectiva)
    y_beiras = pd.DataFrame(data=df_beiras.recolha_selectiva)

    # predicted year to predict the value

    #predicted_year = datetime.today().year + 1

    # split data into Train and Test

    x_train_coimbra, x_test_coimbra, y_train_coimbra, y_test_coimbra = train_test_split(x_df, y_coimbra, test_size = 0.1)
    x_train_viseu, x_test_viseu, y_train_viseu, y_test_viseu = train_test_split(x_df, y_viseu, test_size = 0.1)
    x_train_beiras, x_test_beiras, y_train_beiras, y_test_beiras = train_test_split(x_df, y_beiras, test_size = 0.1)

    # call Linear Regression

    reg_coimbra = LinearRegression()
    reg_viseu = LinearRegression()
    reg_beiras = LinearRegression()

    # fitting the model

    reg_coimbra.fit(x_train_coimbra, y_train_coimbra)
    reg_viseu.fit(x_train_viseu, y_train_viseu)
    reg_beiras.fit(x_train_beiras, y_train_beiras)

    array = np.array(predicted_year)
    float_value = array.astype(np.float32)
    float_value_2D = ([[float_value]])

    # Prediction

    prediction_coimbra = reg_coimbra.predict(float_value_2D)
    prediction_viseu = reg_viseu.predict(float_value_2D)
    prediction_beiras = reg_beiras.predict(float_value_2D)

    predicted_value_coimbra = np.array(prediction_coimbra)
    predicted_value_viseu = np.array(prediction_viseu)
    predicted_value_beiras = np.array(prediction_beiras)

    predicted_value_coimbra = predicted_value_coimbra.item()
    predicted_value_viseu = predicted_value_viseu.item()
    predicted_value_beiras = predicted_value_beiras.item()

    #df2 = pd.DataFrame([[2,3,4]], columns=['A','B','C']) - example of making the pandas dataframe

    recolha_selectiva_coimbra_predicted = pd.DataFrame([[predicted_year,'CIM Coimbra', predicted_value_coimbra]], columns=['year','region','recolha_selectiva'])
    recolha_selectiva_viseu_predicted = pd.DataFrame([[predicted_year,'CIM Viseu', predicted_value_viseu]], columns=['year','region','recolha_selectiva'])
    recolha_selectiva_beiras_predicted = pd.DataFrame([[predicted_year,'CIM Beiras', predicted_value_beiras]], columns=['year','region','recolha_selectiva'])

    #pd.concat([df1, df2], axis=0, sort=False, ignore_index=True) - example of concate two pandas dataframe

    Recolha_selectiva_Coimbra = pd.concat([recolha_selectiva_coimbra, recolha_selectiva_coimbra_predicted], axis=0, sort=False, ignore_index=True) 
    Recolha_selectiva_Viseu = pd.concat([recolha_selectiva_viseu, recolha_selectiva_viseu_predicted], axis=0, sort=False, ignore_index=True) 
    Recolha_selectiva_Beiras = pd.concat([recolha_selectiva_beiras, recolha_selectiva_beiras_predicted], axis=0, sort=False, ignore_index=True) 

    #### === Now plot without prediction  and with prediction values in two seperate line graph to compare === ####

    # Plot line chart for 3-regions without prediction - The recolha_selectiva waste (t) vs. Year

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=recolha_selectiva_coimbra.year, y=recolha_selectiva_coimbra.recolha_selectiva, name='CIM Coimbra', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=recolha_selectiva_viseu.year, y=recolha_selectiva_viseu.recolha_selectiva, name='CIM Viseu', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=recolha_selectiva_beiras.year, y=recolha_selectiva_beiras.recolha_selectiva, name='CIM Beiras', line=dict(color='green', width=3)))
    fig.update_layout(title='Recolha_selectiva Waste in CIM Coimbra, CIM Viseu, CIM Beiras', xaxis_title = 'Year', yaxis_title = 'Recolha_selectiva Waste (t)')

    # Plot line chart for 3-regions with prediction - The Recolha_selectiva waste (t) vs. Year


    percentage_coimbra = percentage(Recolha_selectiva_Coimbra.recolha_selectiva.values[-1], recolha_selectiva_coimbra.recolha_selectiva.values[-1])
    percentage_viseu = percentage(Recolha_selectiva_Viseu.recolha_selectiva.values[-1], recolha_selectiva_viseu.recolha_selectiva.values[-1])
    percentage_beiras = percentage(Recolha_selectiva_Beiras.recolha_selectiva.values[-1], recolha_selectiva_beiras.recolha_selectiva.values[-1])

    fig_total = go.Figure()

    fig_total.add_trace(go.Scatter(x=Recolha_selectiva_Coimbra.year, y=Recolha_selectiva_Coimbra.recolha_selectiva, name=f'CIM Coimbra: grow by "{percentage_coimbra:0.1%}"', line=dict(color='firebrick', width=3)))
    fig_total.add_trace(go.Scatter(x=Recolha_selectiva_Viseu.year, y=Recolha_selectiva_Viseu.recolha_selectiva, name=f'CIM Viseu: grow by "{percentage_viseu:0.1%}"', line=dict(color='royalblue', width=3)))
    fig_total.add_trace(go.Scatter(x=Recolha_selectiva_Beiras.year, y=Recolha_selectiva_Beiras.recolha_selectiva, name=f'CIM Beiras: grow by "{percentage_beiras:0.1%}"', line=dict(color='green', width=3)))
    fig_total.update_layout(title=f'Recolha_selectiva Waste Projections in CIM Coimbra, CIM Viseu, CIM Beiras in "{predicted_year}"', xaxis_title = 'Year', yaxis_title = 'Recolha_selectiva Waste (t)')

    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig, use_container_width=True)

    right_column.plotly_chart(fig_total, use_container_width=True)


