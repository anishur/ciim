import calendar
from datetime import datetime
import streamlit as st
from streamlit_option_menu import option_menu
# import plotly.graph_objects as go
from database import *
import database as db
import pandas as pd
import plotly.express as px
import deta
from deta import Deta
#import socket
#socket.getaddrinfo('localhost', 8503)


def residuos_urbanos_bio():

    regiaos = ["CIM Coimbra", "CIM Viseu", "CIM Beiras"]

    currecy = "percentage (%)"

    currecy_percentage = " (%)"

    st.markdown(
        "<h1 style='text-align: center; color: grey;'>Deposição de residuos urbanos biodegradáveis (RUB) em aterro (%)</h1>",
        unsafe_allow_html=True)

    ##### ======= Drop Down =======######

    years = [datetime.today().year, datetime.today().year - 1, datetime.today().year - 2,
             datetime.today().year - 3, datetime.today().year - 4, datetime.today().year - 5, datetime.today().year - 6]

    months = list(calendar.month_name[1:])

    ###### ============= Navigation =========== #######

    selected = option_menu(menu_title=None,
                           options=["Data Entry", "Data Visualization"],
                           icons=["pencil-fill", "bar-chart-fill"],
                           orientation="horizontal")

    #### ============= This section is for Data Entry ============== ####

    if selected == "Data Entry":

        st.header(f"Data Entry in {currecy}")

        with st.form("entry_form", clear_on_submit=True):

            col1, col2 = st.columns(2)

            col1.selectbox("Select Month: ", months, key="month")

            col2.selectbox("Select Year: ", years, key="year")

            "---"
            with st.expander("Regiao"):

                for regiao in regiaos:

                    st.number_input(f"{regiao}:", min_value=0.00,
                                    format=None, step=10.00, key=regiao)

            with st.expander("Comment"):

                comment = st.text_area(
                    "", placeholder="Enter a comment here ...")

            "---"

            submitted = st.form_submit_button("Save Data")

            if submitted:

                period = f"{st.session_state['year']}_{st.session_state['month']}"

                regiaos = {
                    regiao: st.session_state[regiao] for regiao in regiaos}

                ### ========= Insert into NoSQL database ========== ###

                db.insert_period(period, regiaos, comment)

                st.success("Data Saved!")

    #### ============= This section is for Data Visualization ============== ####

    if selected == "Data Visualization":

        st.header("Data Visualization")

        with st.form("saved_periods"):

            period = st.selectbox("Select Period:", get_all_periods())

            submitted = st.form_submit_button("Plot Period")

            if submitted:

                # get data from NoSQL database

                period_data = db.get_period(period)

                comment = period_data.get("comment")

                regiaos = period_data.get("regiaos")

                # create metrics

                waste_coimbra = regiaos.get("CIM Coimbra")
                waste_viseu = regiaos.get("CIM Viseu")
                waste_beiras = regiaos.get("CIM Beiras")
                total_waste = sum(regiaos.values())

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Total Waste in CIM Coimbra",
                            f"{waste_coimbra} {currecy_percentage}")

                col2.metric("Total Waste in CIM Viseu",
                            f"{waste_viseu} {currecy_percentage}")

                col3.metric("Total Waste in CIM Beiras",
                            f"{waste_beiras} {currecy_percentage}")

                col4.metric("Total Waste Production in 3-Regions",
                            f"{total_waste} {currecy_percentage}")

                st.markdown("""---""")

                st.text(f"Comment {comment}")

                st.markdown("""---""")

                ##### ========== Bar Chart =========== ####

                df1 = pd.DataFrame(
                    {'Regioas': ['CIM Coimbra', 'CIM Beiras', 'CIM Viseu']})
                df2 = pd.DataFrame(
                    [waste_coimbra, waste_beiras, waste_viseu], columns=['Waste'])
                df = pd.concat([df1, df2], axis='columns')

                df1_ = pd.DataFrame(df, columns=['Regioas', 'Waste'])

                fig_1 = px.bar(df1_, y='Regioas', x='Waste', orientation="h", title="<b> Percentage of waste by region </b>",
                               color_discrete_sequence=[
                                   "brown", "green", "blue"],
                               color="Regioas",
                               text_auto=True,
                               template="plotly_white",)

                fig_1.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", xaxis=(dict(showgrid=False)))

                ##### ========== Line Graph =========== ####

                df2_ = df[['Regioas', 'Waste']]

                fig_2 = px.line(
                    df2_, x='Regioas', y='Waste', title='Percentage of waste by Region')

                fig_2.update_layout(
                    xaxis=dict(tickmode="linear"),
                    # plot_bgcolor="rgb(0,0,0,0)",
                    # plot_bgcolor="#FFFFFF",
                    yaxis=(dict(showgrid=False))
                )

                ##### ========== Pie Chart =========== ####

                df['percentage'] = df['Waste']/df['Waste'].sum()

                fig_3 = px.pie(df, values='percentage', names='Regioas', title='Percentage of waste by Region',
                               hole=.3, color_discrete_sequence=px.colors.sequential.RdBu)

                fig_3.update_layout(legend_title="Regioas", legend_y=0.9)

                fig_3.update_traces(textinfo='percent+label',
                                    textposition='inside')

                left_column, center_column, right_column = st.columns(3)

                left_column.plotly_chart(fig_1, use_container_width=True)
                center_column.plotly_chart(fig_2, use_container_width=True)
                right_column.plotly_chart(fig_3, use_container_width=True)
