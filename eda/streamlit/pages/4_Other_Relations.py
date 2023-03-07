import pandas as pd
import streamlit as st

import plotly.express as px

class Keeper:

    def __init__(self):
        self.key = 0

    def new(self):
        res = self.key
        self.key += 1
        return res


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
paired = read_pkl('paired')
concatenated = pd.concat([cloudless, cloudy])

keeper = Keeper()

BAND_PROPERTIES = {
    "Traditional Contrast": "traditional_contrast",
    "RMS Contrast": "rms_contrast",
    "Laplacian Blur": "laplacian_blur",
    "Mean": "mean"
}

PATCH_PROPERTIES = {
    "Saturation": "saturation",
    "Temperature": "temperature",
}

BASIC_ATTR_DICT = {
    "season": ["spring", "summer", "fall", "winter"],
    "type": ["cloudless", "cloudly"],
    "scene": list(map(int, cloudy['scene'].unique()))
}


def specify_property():
    band_property = 'Band property'
    patch_property = 'Patch Property'
    type_property = st.selectbox("Select a type of property", [band_property, patch_property], key=keeper.new())
    if type_property == band_property:
        col1, col2 = st.columns(2)
        with col1:
            band_property = st.selectbox("Select a band", list(range(13)), key=keeper.new())
        with col2:
            measure_property = st.selectbox("Select a property", BAND_PROPERTIES.keys(), key=keeper.new())
        property = "{}_{}".format(band_property, BAND_PROPERTIES[measure_property])
    else:
        property = st.selectbox("Select a property", PATCH_PROPERTIES.keys(), key=keeper.new())
        property = PATCH_PROPERTIES[property]
    return property

def get_full_df_projection(old_df, projection):
    df = pd.DataFrame()
    for attr in projection:
        df1 = pd.DataFrame()
        df1['value'] = old_df[attr]
        df1['measure'] = attr
        df1['band'] = int(attr.split('_')[0])
        for copy_attr in ['type', 'season', 'type']:
            df1[copy_attr] = old_df[copy_attr]
        df = pd.concat([df, df1])
    return df

def comparison_body(df):
    st.header("Compare two properties")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Property A")
        propertyA = specify_property()
    with col2:
        st.subheader("Property B")
        propertyB = specify_property()
    if propertyA == propertyB:
        st.warning(f"Comparing {propertyA} and {propertyB} is trivial since they are the same property.")
    marker_size = st.number_input("Size of the markers", 0.1, 4.0, step=0.1, value=3.0, )
    if hue == "":
        fig = px.scatter(df, x=propertyA, y=propertyB,
                         title="Scatter plot of {} and {}".format(propertyA, propertyB))
    else:
        fig = px.scatter(df, x=propertyA, y=propertyB,
                         title="Scatter plot of {} and {}".format(propertyA, propertyB),
                         color=hue)
    fig.update_traces(marker_size=marker_size)
    st.plotly_chart(fig)

hue = st.sidebar.selectbox("Select a hue", [""] + list(BASIC_ATTR_DICT.keys()))
possible_filters = set(BASIC_ATTR_DICT.keys())
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

comparison_body(concatenated)
