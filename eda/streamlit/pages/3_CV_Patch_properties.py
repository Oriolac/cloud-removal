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


BASIC_ATTR_DICT = {
    "season": ["spring", "summer", "fall", "winter"],
    "type": ["cloudless", "cloudly"],
    "scene": list(map(int, cloudy['scene'].unique()))
}

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
def property_body(df, y, hue):
    if hue != "":
        grouped_df = df[[y, hue]].groupby(by=hue).mean().reset_index()
        fig = px.bar(grouped_df, x=hue, y=y)
        st.write(fig)
    if hue != "":
        fig = px.box(df, y=y, x=hue)
    else:
        fig = px.box(df, y=y)
    st.plotly_chart(fig)



def temperature_body(type_select):
    if len(type_select) == 2:
        df = pd.concat([cloudless, cloudy])
    elif len(type_select) == 1:
        df = DFS[type_select[0]]
    else:
        st.warning("Please select a type in the sidebar")
        return
    st.bar_chart(cloudless, x='season', y='temperature')




def relation_body(df, hue):
    st.markdown(
        f"""
    ## Relation between saturation and temperature
    """
    )
    plot_scatter_relation(df, hue)



def plot_scatter_relation(df, hue_select):
    if hue_select == "":
        fig = px.scatter(df, x='temperature', y='saturation',
                         hover_data=['patch', 'scene', 'season', 'type'])
    else:
        fig = px.scatter(df, x='temperature', y='saturation', color=hue_select,
                     hover_data=['patch', 'scene', 'season', 'type'])
    marker_size = st.number_input("Size of the markers", 0.1, 4.0, step=0.1, value=3.0, )
    fig.update_traces(marker_size=marker_size)
    st.plotly_chart(fig)


concatenated = pd.concat([cloudless, cloudy])
hue = st.sidebar.selectbox("Select a hue", [""] + list(BASIC_ATTR_DICT.keys()))
possible_filters = set(BASIC_ATTR_DICT.keys())
if hue != "":
    possible_filters.remove(hue)
filter_key = st.sidebar.selectbox("Please, select a filter", [""] + list(possible_filters))
if filter_key != "":
    filters = st.sidebar.multiselect("Please, filter by these options", BASIC_ATTR_DICT[filter_key])
    if len(filters) != 0:
        last_shape = concatenated.shape[0]
        concatenated = concatenated[concatenated[filter_key].apply(lambda x: x in list(map(str, filters)))]
        st.sidebar.write(
            "By filtering, it is showing {} rows of {} ({:.2f} %)".format(concatenated.shape[0], last_shape,
                                                                          concatenated.shape[
                                                                              0] / last_shape * 100))
    else:
        st.sidebar.warning("Any filter selected.")
tab1, tab2, tab3 = st.tabs(['Analysis of a band', 'Comparison between bands', 'Comparison between properties'])

saturation_container = st.container()
with saturation_container:
    saturation_container.markdown(
        f"""
    ## Saturation
    """
    )
    property_body(concatenated, 'saturation', hue)
temperature_container = st.container()
with temperature_container:
    temperature_container.markdown(
        f"""
    ## Temperature
    """
    )
    property_body(concatenated, 'temperature', hue)
relation_body(concatenated, hue)
