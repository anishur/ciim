#!/usr/bin/env python
# coding: utf-8
# Build a streamlit web app from scratch (including nosql database + interactive sankey diagram) By: Code Is Fun


import streamlit as st
#import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.express as px
#import matplotlib.colors


theme_plotly = None


def SankeyGraphs():

   ########### ======== Sankey Diagram of Olive Oil ====== ##########

    inputs = {'Azeitona': 6.20, 'Electricidade': 0.24,
              'Propano': 0.02, 'Diesel': 0.028, 'Agua': 1.68, }

    outputs = {'Azeite': 1, 'Folhas': 0.19,
               'Caroco': 0.54, 'Bagaso_Humido': 5.47, 'CO2': 3.26}

    label = list(inputs.keys()) + ["Input"] + list(outputs.keys())

    value = list(inputs.values()) + list(outputs.values())

    source = list(range(len(inputs))) + [len(inputs)] * len(outputs)

    target = [len(inputs)] * len(inputs) + [label.index(output)
                                            for output in outputs.keys()]

    # Data to dict -> dict to sankey diagram

    link = dict(source=source, target=target, value=value)

    node = dict(label=label, pad=20, thickness=30, color="#E694FF")

    data = go.Sankey(link=link, node=node)

    # plot sankey diagram

    fig1 = go.Figure(data)

    title = "Breakdown of Portuguese Olive Oil Production | Sankey Diagram"

    width, height = 700, 500

    fontsize = 18

    fontfamily = 'Time New Roman'

    fig1.update_layout(title_text=title, font_size=fontsize, font_family=fontfamily,
                       width=width, height=height, margin=dict(l=0, r=0, b=20))

    ########### ======== Sankey Diagram of Queijo Ovelha ====== ##########

    # UF = 1 kg queijo ovelha
    # Fronteira Geográfica : Portugal
    # Cardle to Grave

    # Inputs

    inputs2 = {'Leite de ovelha': 5.5, 'Sal': 0.04,
               'Hipoclorito de Sódio': 4.55, 'Ácido Nítrico': 4.55,
               'Electricidade': 0.589, 'Propano': 0.033, 'Agua': 0.595,
               'Cartão/Etiquetas': 0.003, }

    outputs2 = {'Queijo de ovelha': 1, 'CO2 (Combustion)': 0.0846,
                'CH4 (Combustion)': 0.00000134, 'N2O (Combustion)': 0.000000134,
                'NOx (Combustion)': 0.000000099, 'CO (Combustion)': 0.000000039,
                'NMVOCs (Combustion)': 0.000000031, 'SOx (Combustion)': 0.000000000000001,
                }
    label2 = list(inputs2.keys()) + ["Input"] + list(outputs2.keys())

    value2 = list(inputs2.values()) + list(outputs2.values())

    source2 = list(range(len(inputs2))) + [len(inputs2)] * len(outputs2)

    target2 = [len(inputs2)] * len(inputs2) + [label2.index(output2)
                                               for output2 in outputs2.keys()]

    # Data to dict -> dict to sankey diagram

    link2 = dict(source=source2, target=target2, value=value2)

    node2 = dict(label=label2, pad=20, thickness=30, color="#E694FF")

    data2 = go.Sankey(link=link2, node=node2)

    fig2 = go.Figure(data2)

    title2 = "Breakdown of Portuguese Queijo Ovelha Production | Sankey Diagram"

    width, height = 700, 500

    fontsize = 18

    fontfamily = 'Time New Roman'

    fig2.update_layout(title_text=title2, font_size=fontsize, font_family=fontfamily,
                       width=width, height=height, margin=dict(l=0, r=0, b=20))

    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig1, use_container_width=True)
    right_column.plotly_chart(fig2, use_container_width=True)


def SankeyGraphs2():

    ########### ======== Sankey Diagram of Queijo Ovelha ====== ##########

    # UF = 1 kg queijo de vaca curado
    # Fronteira Geográfica : Portugal
    # Cardle to Gate

    # Inputs de Materiais

    inputs3 = {'Leite de vaca': 8.24, 'Leite em pó': 0.129, 'Agua': 23,
               'Ácido Nítrico': 0.037, 'Hidróxido de Sódio': 0.101, 'Cartão': 0.115,
               'HDPE': 0.015, 'Fuel Oil': 0.401, 'Electricidade': 1.08,
               }

    outputs3 = {'Queijo curado': 1, 'Whey em pó': 0.49, 'Cartão para reciclagem': 0.006,
                'Plásticos para reciclagem': 0.000075, 'Cartão para incineração': 0.0012,
                'Plásticos para incineração': 0.000045, 'Lamas para incineração': 0.199,
                'Resíduos para compostagem': 0.000552, 'Cartão para aterro': 0.00427,
                'Plásticos para aterro': 0.000159, 'Monóxido de Carbono': 0.00663,
                'Óxidos de azoto': 0.00463, 'Partículas': 0.01802, 'Dióxido de enxofre': 0.00566,
                'Substâncias orgânicas': 0.000249, 'Carência Química de Oxigénio': 0.08352,
                'Sólidos Suspensos': 0.01768, 'Azoto total': 0.00212, 'Fósforo': 0.00142,
                'Nitrato': 0.000074, 'Óleos': 0.0045,
                }

    label3 = list(inputs3.keys()) + ["Input"] + list(outputs3.keys())

    value3 = list(inputs3.values()) + list(outputs3.values())

    source3 = list(range(len(inputs3))) + [len(inputs3)] * len(outputs3)

    target3 = [len(inputs3)] * len(inputs3) + [label3.index(output3)
                                               for output3 in outputs3.keys()]

    # Data to dict -> dict to sankey diagram

    link3 = dict(source=source3, target=target3, value=value3)

    node3 = dict(label=label3, pad=20, thickness=30, color="#E694FF")

    data3 = go.Sankey(link=link3, node=node3)

    fig3 = go.Figure(data3)

    title3 = "Breakdown of Portuguese queijo de vaca curado Production | Sankey Diagram"

    width, height = 700, 500

    fontsize = 18

    fontfamily = 'Time New Roman'

    fig3.update_layout(title_text=title3, font_size=fontsize, font_family=fontfamily,
                       width=width, height=height, margin=dict(l=0, r=0, b=20))

    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig3, use_container_width=True)

    ################# previous plot of sankey diagram ########################

    #df = pd.read_excel("data/azeite.xlsx")

    # def SankeyGraphs():

    #    cols = ["Region", "Segment", "Category", "Sub-Category"]

    #    value = "Sales"

    #    value_suffix = " Lit"

    #    title = "Breakdown of Portuguese Olive Oil Production | Sankey Diagram"
    #    width, height = 700, 500
    #    fontsize = 24
    #    fontfamily = 'Time New Roman'
    #    bgcolor = 'Black'
    #    link_capacity = 0.3
    #    node_colors = px.colors.qualitative.D3
    #    s = []
    #    t = []
    #    v = []
    #    labels = np.unique(df[cols].values)

#   for i in range(len(cols) - 1):
#        s.extend(df[cols[i]].tolist())
#        t.extend(df[cols[i + 1]].tolist())
#        v.extend(df[value].tolist())

#    links = pd.DataFrame({"source": s, "target": t, "value": v})

#    links = links.groupby(["source", "target"],
#                          as_index=False).agg({"value": "sum"})

# save excel file
# links.to_excel("links1.xlsx")

#    links = pd.read_excel('data/links1.xlsx')

#    colors = [matplotlib.colors.to_rgb(i) for i in node_colors]

#    label_colors, links["link_c"] = [], 0

#    c, max_colors = 0, len(colors)

#    for l in range(len(labels)):

#        label_colors.append(colors[c])

#        link_color = colors[c] + (link_capacity,)

#        links.loc[links.source == labels[l], [
#            "link_c"]] = "rgba" + str(link_color)

#        links = links.replace({labels[l]: l})

#        if c == max_colors - 1:

#            c = 0

#        else:

#            c += 1

#    label_colors = ["rgb" + str(i) for i in label_colors]

#    fig = go.Figure(
#        data=[
#            go.Sankey(
#                valuesuffix=value_suffix,
#                node=dict(label=labels, color=label_colors),
#                link=dict(
#                    source=links["source"],
#                    target=links["target"],
#                    value=links["value"],
#                    color=links["link_c"],
#                ),
#            )
#        ]
#    )

#    fig.update_layout(title_text=title, font_size=fontsize, font_family=fontfamily, width=width, height=height,
#                      paper_bgcolor=bgcolor, title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"})


##left_column, center_column, right_column = st.columns(3)
"""
    left_column, right_column = st.columns(2)

    left_column.plotly_chart(fig, use_container_width=True)

    ## center_column.plotly_chart(fig, use_container_width=True)

    # right_column.plotly_chart(
    # fig, use_container_width = True, theme = theme_plotly)

"""
