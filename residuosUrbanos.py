import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
import pymysql
import mysql


querystr = "SELECT * FROM waste_collected"

engine = create_engine(
    'mysql+pymysql://root:1234@127.0.0.1:3300/db_urban_waste').connect()

df = pd.read_sql_query(querystr, engine)

df = df.iloc[:, 1:]


def residuosUrbanos():

    with st.expander("Regional Statistics - Resíduos urbanos recolhidos"):
        showdata = st.multiselect('Filter: ', df.columns, default=[])
        st.dataframe(df[showdata], use_container_width=True)

        # compute top analytics

        total_waste = float(df['total'].sum())
        total_papel = float(df['papel'].sum())
        total_plastico = float(df['plastico'].sum())
        total_metal = float(df['metal'].sum())
        total_vidro = float(df['vidro'].sum())

        total1, total2, total3, total4, total5 = st.columns(5, gap='small')

        #### 6-10 ####
        total_madeira = float(df['madeira'].sum())
        total_equipamentos = float(df['equipamentos'].sum())
        total_pilhas = float(df['pilhas'].sum())
        total_oleos_alimentares = float(df['oleos_alimentares'].sum())
        total_outros = float(df['outros'].sum())

        total6, total7, total8, total9, total10 = st.columns(5, gap='small')

        with total1:
            st.info('Waste', icon='🏭')
            st.metric(label="Total Waste", value=f"{total_waste: .0f}")

        with total2:
            st.info('Papel', icon='🏢')
            st.metric(label="Total Papel Waste",
                      value=f"{total_papel: .0f}")

        with total3:
            st.info('Plastico', icon='🏠')
            st.metric(label="Total Plastico Waste",
                      value=f"{total_plastico: .0f}")

        with total4:
            st.info('Metal', icon='🟩')
            st.metric(label="Total Metal Waste", value=f"{total_metal: .0f}")

        with total5:
            st.info('Vidro', icon='🚮')
            st.metric(label="Total Vidro Waste", value=f"{total_vidro: .0f}")

        with total6:
            st.info('Madeira', icon='🏭')
            st.metric(label="Total Madeira Waste",
                      value=f"{total_madeira: .0f}")

        with total7:
            st.info('Equipamentos', icon='🏭')
            st.metric(label="Total Equipamentos Waste",
                      value=f"{total_equipamentos: .0f}")

        with total8:
            st.info('Pilhas', icon='🏢')
            st.metric(label="Total Pilhas Waste",
                      value=f"{total_pilhas: .0f}")

        with total9:
            st.info('Oleos_Alimentares', icon='🏠')
            st.metric(label="Total Oleos_alimentares Waste",
                      value=f"{total_oleos_alimentares: .0f}")

        with total10:
            st.info('Outros', icon='🟩')
            st.metric(label="Total Outros Waste", value=f"{total_outros: .0f}")

        st.markdown("""---""")
