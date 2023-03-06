import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt


def read_pkl(key_pkl):
    if key_pkl not in st.session_state:
        paired: pd.DataFrame = pd.concat([pd.read_pickle(f'eda/streamlit/data/{key_pkl}_{s}.pkl') for s in range(3)])
        st.session_state[key_pkl] = paired
    else:
        paired = st.session_state[key_pkl]
    paired['type'] = key_pkl
    return paired

cloudless = read_pkl('cloudless')
cloudy = read_pkl('cloudy')

class Keeper:

    def __init__(self):
        self.key = 0

    def new(self):
        res = self.key
        self.key += 1
        return res


keeper = Keeper()

st.markdown(
    f"""
    # Patch properties
    """
)

DFS = {
    "Cloudless": cloudless,
    "Cloudy": cloudy,
}

@st.cache_data
def property_body(type_select, y):
    if len(type_select) == 2:
        df = pd.concat([cloudless, cloudy])
    elif len(type_select) == 1:
        df = DFS[type_select[0]]
    else:
        st.warning("Please select a type in the sidebar")
        return
    grouped_df = df.groupby(by='season').mean().reset_index()
    fig = px.bar(grouped_df, x='season', y=y)
    st.write(fig)
    fig = px.box(df, y=y, x='season')
    st.plotly_chart(fig)

saturation_container = st.container()
with saturation_container:
    saturation_container.markdown(
        f"""
    ## Saturation
    """
    )
    type_select = st.multiselect("Select a type", ['Cloudless', 'Cloudy'], key=keeper.new())
    property_body(type_select, 'saturation')


def temperature_body(type_select):
    if len(type_select) == 2:
        df = pd.concat([cloudless, cloudy])
    elif len(type_select) == 1:
        df = DFS[type_select[0]]
    else:
        st.warning("Please select a type in the sidebar")
        return
    st.bar_chart(cloudless, x='season', y='temperature')

temperature_container = st.container()
with temperature_container:
    temperature_container.markdown(
        f"""
    # Temperature
    """
    )
    type_select = temperature_container.multiselect("Select a type", ['Cloudless', 'Cloudy'], key=keeper.new())
    property_body(type_select, 'temperature')


def relation_body():
    st.markdown(
        f"""
    # Relation
    """
    )
    hue_select = st.selectbox("Select a hue", ['season', 'type'], key=keeper.new())
    if hue_select == 'season':
        filter_select = st.multiselect("Select a type", ['Cloudless', 'Cloudy'])
        if len(filter_select) == 2:
            df = pd.concat([cloudless, cloudy])
        elif len(type_select) == 1:
            df = DFS[type_select[0]]
        else:
            st.warning("Please select a type.")
            return
    else:
        filter_select = st.multiselect("Select a season", ['spring', 'summer', 'fall', 'winter'])
        df = pd.concat([cloudless, cloudy])
        if len(filter_select) > 0:
            df = df[df['season'].apply(lambda x: x in filter_select)]
        else:
            st.warning("Please select a season.")
            return
    plot_scatter_relation(df, hue_select)



def plot_scatter_relation(df, hue_select):
    fig = px.scatter(df, x='temperature', y='saturation', color=hue_select,
                     hover_data=['patch', 'scene', 'season', 'type'])
    marker_size = st.number_input("Size of the markers", 0.1, 4.0, step=0.1, value=3.0, )
    fig.update_traces(marker_size=marker_size)
    st.plotly_chart(fig)


relation_body()
