import streamlit as st
import datetime as dt
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
#import plotly.graph_objects as go


def recolhidos():

    # st.set_page_config(layout="wide")

    theme_plotly = None

    st.header("Resíduos urbanos recolhidos (t) por Localização geográfica")

    display_text = "Período de referência dos dados"

    #now = dt.datetime.now().strftime('%y-%m-%d %H:%M')
    #st.write(f'It is now {now} and {display_text}')

    st.write(f'{display_text}')

    #original_list = ['Select Year', '2020', '2019', '2018', '2017']

    # years = ['Select Year', dt.datetime.today().year, dt.datetime.today().year-1, dt.datetime.today().year-2, dt.datetime.today().year-3, dt.datetime.today().year-4,
    #         dt.datetime.today().year-5, dt.datetime.today().year-6, dt.datetime.today().year-7, dt.datetime.today().year-8, dt.datetime.today().year-9, ]

    years = [dt.datetime.today().year, dt.datetime.today().year-1, dt.datetime.today().year-2, dt.datetime.today().year-3, dt.datetime.today().year-4,
             dt.datetime.today().year-5, dt.datetime.today().year-6, dt.datetime.today().year-7, dt.datetime.today().year-8, dt.datetime.today().year-9, ]

    #result = st.selectbox('Select year for data visualization', original_list)
    result = st.selectbox('Select year for data visualization', years)

    #st.write(f'You have selected Year : {result}')

    #### ====== connection to MySQL database ======= #####

    querystr = "SELECT * FROM waste_collected_by_year"

    engine = create_engine(
        'mysql+pymysql://root:1234@127.0.0.1:3300/db_urban_waste_by_year').connect()

    df = pd.read_sql_query(querystr, engine)

    #st.header("Data Visualization")

    # if df.loc[df['year'] == result]:

    #    pass

    # else:

    #    print("Data is not available!")

    df = df.loc[df['year'] == result]

    # compute top analytics

    total_waste = float(df['total'].sum())
    total_papel = float(df['papel'].sum())
    total_plastico = float(df['plastico'].sum())
    total_metal = float(df['metal'].sum())
    total_vidro = float(df['vidro'].sum())
    total_madeira = float(df['madeira'].sum())
    total1, total2, total3, total4, total5, total6 = st.columns(6, gap='small')

    #### 7-12 ####

    total_equipamentos = float(df['equipamentos'].sum())
    total_pilhas = float(df['pilhas'].sum())
    total_oleos_alimentares = float(df['oleos_alimentares'].sum())
    total_outros = float(df['outros'].sum())
    total_recolha_indiferenciada = float(df['recolha_indiferenciada'].sum())
    total_recolha_selectiva = float(df['recolha_selectiva'].sum())

    total7, total8, total9, total10, total11, total12 = st.columns(
        6, gap='small')

    # total1, total2, total3, total4, total5, total6, total7, total8, total9, total10 = #st.columns(10, gap='small') #gap='large')

    with total1:
        st.info('Waste', icon='🏭')
        st.metric(label="Total Waste (t)", value=f"{total_waste: .0f}")

    with total2:
        st.info('Papel', icon='🏢')
        st.metric(label="Total Papel Waste (t)",
                  value=f"{total_papel: .0f}")

    with total3:
        st.info('Plastico', icon='🏠')
        st.metric(label="Total Plastico Waste (t)",
                  value=f"{total_plastico: .0f}")

    with total4:
        st.info('Metal', icon='🟩')
        st.metric(label="Total Metal Waste (t)", value=f"{total_metal: .0f}")

    with total5:
        st.info('Vidro', icon='🚮')
        st.metric(label="Total Vidro Waste (t)", value=f"{total_vidro: .0f}")

    with total6:
        st.info('Madeira', icon='🏭')
        st.metric(label="Total Madeira Waste (t)",
                  value=f"{total_madeira: .0f}")

    with total7:
        st.info('Equipamentos', icon='🏭')
        st.metric(label="Total Equipamentos Waste (t)",
                  value=f"{total_equipamentos: .0f}")

    with total8:
        st.info('Pilhas', icon='🏢')
        st.metric(label="Total Pilhas Waste (t)",
                  value=f"{total_pilhas: .0f}")

    with total9:
        st.info('Oleos_Alimentares', icon='🏠')
        st.metric(label="Total Oleos_alimentares Waste (t)",
                  value=f"{total_oleos_alimentares: .0f}")

    with total10:
        st.info('Outros', icon='🟩')
        st.metric(label="Total Outros Waste (t)", value=f"{total_outros: .0f}")

    with total11:
        st.info('Recolha Indiferenciada', icon='🏢')
        st.metric(label="Total Recolha Indiferenciada (t)",
                  value=f"{total_recolha_indiferenciada: .0f}")

    with total12:
        st.info('Recolha Selectiva', icon='🚮')
        st.metric(label="Total Recolha Selectiva (t)",
                  value=f"{total_recolha_selectiva: .0f}")

    st.markdown("""---""")

    ##### Dataframe for all graphs below #####

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
    columns_indiferenciada = ['region', 'recolha_indiferenciada']
    columns_selectiva = ['region', 'recolha_selectiva']

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
    df11 = pd.DataFrame(df, columns=columns_indiferenciada)
    df12 = pd.DataFrame(df, columns=columns_selectiva)

    ################### 1st row graph groups start from here ###########################

    # 1st Bar graph

    #fig_1 = px.bar(df1, x='region', y='total')

    fig_1 = px.bar(
        df1,
        x="total",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="Total Waste by Region in Year  " + str(result),
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

    #title_lines = "Resíduos urbanos recolhidos no ano de ",

    fig_residuos_urbanos = px.line(
        residuos_urbanos, x='region', y=residuos_urbanos.columns[1:11], title="Resíduos urbanos recolhidos no ano de "+str(result),)

    # 3rd Pie Chart

    df1['percentage'] = df1['total']/df1['total'].sum()

    fig_pie = px.pie(df1, values='percentage', names='region', title='Municipal solid waste (%) in year '+str(result),
                     hole=.3, color_discrete_sequence=px.colors.sequential.RdBu)

    fig_pie.update_layout(legend_title="Region", legend_y=0.9)

    fig_pie.update_traces(textinfo='percent+label', textposition='inside')

    left_column, center_column, right_column = st.columns(3)

    left_column.plotly_chart(fig_1, use_container_width=True)

    center_column.plotly_chart(fig_residuos_urbanos, use_container_width=True)

    right_column.plotly_chart(
        fig_pie, use_container_width=True, theme=theme_plotly)

    st.markdown("""---""")

    ################### 2nd row graph groups start from here ###########################

    # 2nd Bar graph

    #fig_2 = px.bar(df2, x='region',y='papel')

    fig_2 = px.bar(
        df2,
        x="papel",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Paper Waste by Region </b>"+str(result),
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
        title="<b> Total Plastic Waste by Region </b>"+str(result),
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
        title="<b> Total Metal Waste by Region </b>"+str(result),
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

    st.markdown("""------""")

    ################### 3rd row graph groups start from here ###########################

    # 5th Bar graph
    #fig_5 = px.bar(df5, x='region',y='vidro')

    fig_5 = px.bar(
        df5,
        x="vidro",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Glass Waste by Region </b>"+str(result),
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
        title="<b> Total Wood Waste by Region </b>"+str(result),
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
        title="<b> Total Equipments Waste by Region </b>"+str(result),
        color_discrete_sequence=["brown", "green", "blue"],
        color="region",
        text_auto=True,
        template="plotly_white",
    )

    fig_7.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    left_column, center_column, right_column = st.columns(3)

    left_column.plotly_chart(fig_5, use_container_width=True)
    center_column.plotly_chart(fig_6, use_container_width=True)
    right_column.plotly_chart(fig_7, use_container_width=True)

    st.markdown("""------""")

    ################### 4th row graph groups start from here ###########################

    # 8th Bar graph
    # fig_8 = px.bar(df8, x='region',y='pilhas')

    fig_8 = px.bar(
        df8,
        x="pilhas",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Battery Waste by Region </b>"+str(result),
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
        title="<b> Total Oleos alimentares Waste by Region </b>"+str(result),
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
        title="<b> Total Miscellaneous Waste by Region </b>"+str(result),
        color_discrete_sequence=["brown", "green", "blue"],
        color="region",
        text_auto=True,
        template="plotly_white",
    )

    fig_10.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    left_column, center_column, right_column = st.columns(3)

    left_column.plotly_chart(fig_8, use_container_width=True)
    center_column.plotly_chart(fig_9, use_container_width=True)
    right_column.plotly_chart(fig_10, use_container_width=True)

    st.markdown("""------""")

################### 5th row graph groups start from here ###########################

    # 11th Bar graph

    fig_11 = px.bar(
        df11,
        x="recolha_indiferenciada",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Recolha Indiferenciada by Region </b>"+str(result),
        color_discrete_sequence=["brown", "green", "blue"],
        color="region",
        text_auto=True,
        template="plotly_white",
    )

    fig_11.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # 12th Bar graph
    # fig_12 = px.bar(df12, x='region',y='oleos_alimentares')

    fig_12 = px.bar(
        df12,
        x="recolha_selectiva",
        # y=compaines_by_region.index,
        y="region",
        orientation="h",
        title="<b> Total Recolha Selectiva by Region </b>"+str(result),
        color_discrete_sequence=["brown", "green", "blue"],
        color="region",
        text_auto=True,
        template="plotly_white",
    )

    fig_12.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    left_column, center_column, right_column = st.columns(3)

    left_column.plotly_chart(fig_11, use_container_width=True)
    center_column.plotly_chart(fig_12, use_container_width=True)
    #right_column.plotly_chart(fig_10, use_container_width=True)
