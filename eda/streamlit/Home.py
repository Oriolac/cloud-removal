import streamlit as st
import pandas as pd
import logging
import plotly.express as px

st.write("# Exploratory Data Analysis")

st.markdown(
    """
    EDA stands for Exploratory Data Analysis. It is a process of analyzing and summarizing data sets to gain insights
    into the data and understand its main characteristics. The goal of EDA is to identify patterns, relationships,
    and trends in the data that can be used to inform further analysis or decision-making.
    It helps to identify outliers, missing data, and other anomalies that may affect the analysis.
    
    By performing EDA, analysts can also get a better sense of the underlying distribution of the data, which can
    help to guide subsequent analysis and modeling efforts. Overall, EDA is an important step in any data analysis
    project and can help to ensure that the analysis is based on accurate and reliable data.
    
"""
)

def read_pkl(key_pkl):
    logging.info(f"Getting {key_pkl}")
    if key_pkl not in st.session_state:
        paired: pd.DataFrame = pd.concat([pd.read_pickle(f'eda/streamlit/data/{key_pkl}_{s}.pkl') for s in range(3)])
        st.session_state[key_pkl] = paired
    else:
        paired = st.session_state[key_pkl]
    paired['type'] = key_pkl
    logging.info(f"Extracted {key_pkl}")
    return paired

cloudless = read_pkl('cloudless')
cloudy = read_pkl('cloudy')
paired = read_pkl('paired')

df = pd.concat([cloudless, cloudy])

st.markdown(
    f"""
    ## Dataset
    There is a total of **{df.shape[0]}** images: **{cloudless.shape[0]}** cloudless images and
     **{cloudy.shape[0]}** cloudy images. Each image is called **patch**.
     
    There are **{paired['scene'].unique().shape[0]}** scenes.
    """
)

option = st.selectbox("How would you like to group each patch?", ('season', 'roi', 'scene'))


df_groupby = df.groupby(by=[option]).count()
st.bar_chart(df_groupby['patch'])


counter_patch = df.groupby(by=option).count()
fig = px.histogram(counter_patch, x="patch")
st.plotly_chart(fig, theme="streamlit",)

