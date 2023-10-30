

from datetime import datetime
from database import *
from residuosUrbanosBio_RUB import *
from residuosUrbanos import residuosUrbanos
from residuosUrbanosRecolhidos import recolhidos
from Sankey import *
from prediction import *
from graphs2 import *
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import streamlit as st
#import warnings
# warnings.filterwarnings("ignore")

#import shutup

# shutup.are_warnings_muted()
# .please()


# page behavior
# https://www.webfx.com/tools/emoji-cheat-sheet/

def set_page():

    st.set_page_config(page_title="CiiM",
                       page_icon="images\ciim_logo_final.png", layout="wide")
    st.subheader("  :blue[CiiM BUSINESS ANALYTICS DASHBOARD]📶 ")
    st.markdown("##")


set_page()

# remove default theme

theme_plotly = None

# Style

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# read data

df = pd.read_excel('data\info_res_urbanos.xlsx', sheet_name='Sheet1')


# side bar


st.sidebar.image(
    "images\ciim_logo_final.png", use_column_width=True)

# switcher

st.sidebar.header("Please seclect filter")

region = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique(),
)

# DataFrame Query

df_selection = df.query("Region==@region")


# show Excel Workbook in the Dashboard


def Home():

    with st.expander("Regional Statistics"):
        showdata = st.multiselect('Filter: ', df_selection.columns, default=[])
        st.dataframe(df_selection[showdata], use_container_width=True)

    # compute top analytics

    total_companies = float(df_selection['Number of companies'].sum())
    total_largecompanies = float(df_selection['Large companies'].sum())
    total_inhabitants = float(df_selection['Inhabitans'].sum())
    total_areas = float(df_selection['Area km sq'].sum())
    total_residuos = float(
        df_selection['Residuos urbanos kg por habitantes 2020'].sum())

    total1, total2, total3, total4, total5 = st.columns(5, gap='large')

    with total1:
        st.info('Companies', icon='🏭')
        st.metric(label="Total Companies", value=f"{total_companies: .0f}")

    with total2:
        st.info('Large Companies', icon='🏢')
        st.metric(label="Total Large Companies",
                  value=f"{total_largecompanies: .0f}")

    with total3:
        st.info('Inhabitants', icon='🏠')
        st.metric(label="Total Inhabitants", value=f"{total_inhabitants: .0f}")

    with total4:
        st.info('Areas', icon='🟩')
        st.metric(label="Total Areas km sq", value=f"{total_areas: .0f}")

    with total5:
        st.info('Total Residuos urbanos', icon='🚮')
        st.metric(label="kg por habitant em 2020",
                  value=f"{total_residuos: .0f}")

    st.markdown("""---""")


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

    x = datetime.today().year + 1  # next year taken from system
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

    fig_pred.update_layout(title='Prediction of Residuos Urbanos kg por habitantes',
                           xaxis_title='Year', yaxis_title='Residuos Urbanos kg por habitantes')

    left_column, center_column, right_column = st.columns(3)

    left_column.plotly_chart(fig, use_container_width=True)

    center_column.plotly_chart(fig_pred, use_container_width=True)

    #right_column.plotly_chart(fig_percentage2020, use_container_width=True)


def sideBar():

    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["Home", "Residuos Urbanos", "Resíduos Urbanos Recolhidos",
                     "Residuos Urbanos Biodegrdáveis(RUB)", "Sankey Diagram", "Prediction"],
            icons=["house", "eye", "eye", "eye", "eye", "eye"],
            menu_icon="cast",
            default_index=0
        )
    if selected == "Home":

        # try:
        Home()
        Graphs1()
        # Graphs1()

        # except:
        #    st.warning("one or more options are mandatory! ")

    if selected == "Residuos Urbanos":
        # try:
        residuosUrbanos()
        Graphs2()

        # except:
        #    st.warning("one or more options are mandatory! ")

    if selected == "Resíduos Urbanos Recolhidos":
        # try:
        recolhidos()

        # except:
        #    st.warning("one or more options are mandatory! ")

    if selected == "Residuos Urbanos Biodegrdáveis(RUB)":

        residuos_urbanos_bio()
        # try:

        # residuos_urbanos_bio()

        # except:
        ##    st.warning("one or more options are mandatory! ")

    if selected == "Sankey Diagram":
        try:
            # SankeyDiagram()
            SankeyGraphs()
            SankeyGraphs2()

        except:
            st.warning("one or more options are mandatory! ")

    if selected == "Prediction":
        # try:
        display_header()
        prediction_total_waste()
        prediction_papel_waste()
        prediction_plastico_waste()
        prediction_metal_waste()
        prediction_vidro_waste()
        prediction_madeira_waste()
        prediction_equipamentos_waste()
        prediction_pilhas_waste()
        prediction_oleos_alimentares_waste()
        prediction_outros_waste()
        prediction_recolha_indiferenciada_waste()
        prediction_recolha_selectiva_waste()


        # except:
        #    st.warning("one or more options are mandatory! ")

    st.sidebar.image(
        "images\CECOLAB_logo_final.png", caption="", use_column_width=True)
    st.sidebar.image(
        "images\europa_logo_final.png", caption="", use_column_width=True)
    st.sidebar.image(
        "images\Versao1_cores.png", caption="", use_column_width=True)


sideBar()

############# ==========configure sidebar width==========############


def config_sidebar():

    config_sidebar_width = '''
        <style>
            section[data-testid="stSidebar"] .css-ng1t4o {{width: 12rem;}}
            section[data-testid="stSidebar"] .css-1d391kg {{width: 12rem;}}
        </style>
    '''
    st.markdown(config_sidebar_width, unsafe_allow_html=True)


config_sidebar()
############## ===============Hide the Menu from application===========#######


def hide_menu():

    hide_menu_style = """
            <style>
            #MainMenu {
                visibility: visible;
                //visibility: hidden;
                }

            footer{
                visibility:hidden;
                }
            footer:after{
                    visibility:hidden;
                    content: 'Copyright @ 2023: CECOLAB';
                    display:block;
                    position:relative;
                    color:tomato;
                    padding:5px;
                    top:3px;
                    text-align: right;
                }

            </style>

    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)


hide_menu()
############## ==================Footer================##############


def footer():

    footer = """
    <style>
    .footer {
    
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    
    background-color: rgba(76,76,76,0.8);
    
    color: tomato;
    text-align: center;
    }

    </style>

    <div class="footer">
    <p>Developed by <a href="https://www.cecolab.pt/">CECOLAB</a><br /><a href="https://www.cecolab.pt/">Copyright @ 2023: CECOLAB</a></p>
    </div> 
    """

    st.markdown(footer, unsafe_allow_html=True)


footer()
