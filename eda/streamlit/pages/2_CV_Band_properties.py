import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
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
paired = read_pkl('paired')

BANDS = [f"{b}" for b in range(13)]


PROPERTIES_DICT = {
    "Traditional Contrast": "traditional_contrast",
    "Michelson Contrast": "michelson_contrast",
    "RMS Contrast": "rms_contrast",
    "Laplacian Blur": "laplacian_blur",
}

BASIC_ATTR_DICT = {
    "season": ["spring", "summer", "fall", "winter"],
    "type": ["cloudless", "cloudly"],
}

NAME_PROPERTIES = list(PROPERTIES_DICT.keys())


def get_full_df_projection(old_df, projection):
    df = pd.DataFrame()
    for attr in projection:
        df1 = pd.DataFrame()
        df1['value'] = old_df[attr]
        df1['band'] = int(attr.split('_')[0])
        for copy_attr in ['type', 'season', 'type']:
            df1[copy_attr] = old_df[copy_attr]
        df = pd.concat([df, df1])
    return df


def band_analysis(df):
    analysis_cont = st.container()
    with analysis_cont:
        col1, col2 = st.columns(2)
        band = col1.selectbox("Please, choose a band", BANDS)
        projection = "{band}_{property}".format(band=band, property=PROPERTIES_DICT[select_property])
        df = df[[projection, 'type', 'season']]
        fig = px.box(df, y=projection)
        st.plotly_chart(fig)

        hue = st.selectbox("Please, select a hue", BASIC_ATTR_DICT.keys())
        for key in BASIC_ATTR_DICT.keys():
            if key != hue:
                filter_key = key
                break
        filters = st.multiselect("Please, select a filter", BASIC_ATTR_DICT[filter_key])
        df = df[df[filter_key].apply(lambda x: x in filters)]
        fig = px.histogram(df, x=projection, color=hue)
        st.plotly_chart(fig, theme="streamlit")


def get_filter(df, selected_property, container):
    df = df.groupby(by=['band', 'type']).mean().reset_index()
    df.sort_values(by='band')
    fig, ax = plt.subplots(1, 1)
    ax.set_title(selected_property + " by mean")
    sns.barplot(df, x='band', y='value', hue='type', ax=ax)
    container.write(fig)


select_property = st.sidebar.selectbox("Select a property", NAME_PROPERTIES)


def comparing_bands(df):
    bands_container = st.container()
    select_all_bands = st.checkbox("Select all bands")

    if select_all_bands:
        selected_bands = bands_container.multiselect("Select bands", BANDS, BANDS, disabled=True)
    else:
        selected_bands = bands_container.multiselect("Select bands", BANDS)
    properties_container = st.container()

    if len(selected_bands) == 0:
        properties_container.warning("You must select one band and one property at least")
        return
    projection = [
        "{band}_{property}".format(band=band, property=PROPERTIES_DICT[select_property]) for band in selected_bands
    ]
    df = get_full_df_projection(df, projection)
    get_filter(df, select_property, properties_container)
    st.markdown(
        """
        ### Box plot
        """
    )
    show_boxplot = st.checkbox("Show box plot")
    if show_boxplot:
        st.warning("Due to the cost of the box plot, it is recommended to sample.")
        samples = st.slider("Select % of samples", 1, 100)
        samples_int = int(df.shape[0] * samples / 100)
        st.markdown(
            f"The sample is composed by {samples_int} rows ({samples_int // 13} patches) from the former {df.shape[0]} rows ({df.shape[0] // 13} patches)")

        fig = px.box(df.sample(samples_int), x='band', y='value')
        st.plotly_chart(fig)
    st.markdown(
        """
        ### Histograms
        """
    )
    show_histograms = st.checkbox("Show histograms")
    if show_histograms:
        hue = st.selectbox("Select a hue", [""] + list(BASIC_ATTR_DICT.keys()))
        for b in selected_bands:
            fig = px.histogram(df[df.band == int(b)],
                               title=f"Histogram of {select_property} in band {b}",
                               x="value", color=hue if hue != "" else None)
            st.plotly_chart(fig, theme="streamlit", )


concatenated = pd.concat([cloudless, cloudy])

tab1, tab2, tab3 = st.tabs(['Analysis of a band', 'Comparison between bands', 'Comparison between properties'])

with tab1:
    st.markdown("""
        ## Analysing a band
    """)
    band_analysis(concatenated)

with tab2:
    st.markdown("""
        ## Comparing bands
    """)
    comparing_bands(concatenated)

with tab3:
    st.markdown("""
        ## Comparing properties
    """)
