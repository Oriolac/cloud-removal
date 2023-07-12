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
    "RMS Contrast": "rms_contrast",
    "Laplacian Blur": "laplacian_blur",
}

BASIC_ATTR_DICT = {
    "season": ["spring", "summer", "fall", "winter"],
    "type": ["cloudless", "cloudly"],
    "scene": list(map(int, cloudy['scene'].unique()))
}

NAME_PROPERTIES = list(PROPERTIES_DICT.keys())


def get_full_df_projection(old_df, projection):
    df = pd.DataFrame()
    for attr in projection:
        df1 = pd.DataFrame()
        df1['value'] = old_df[attr]
        df1['measure'] = attr
        df1['band'] = int(attr.split('_')[0])
        for copy_attr in BASIC_ATTR_DICT.keys():
            df1[copy_attr] = old_df[copy_attr]
        df = pd.concat([df, df1])
    return df


def band_analysis(df):
    analysis_cont = st.container()
    with analysis_cont:
        col1, col2 = st.columns(2)
        band = col1.selectbox("Please, choose a band", BANDS)
        projection = "{band}_{property}".format(band=band, property=PROPERTIES_DICT[select_property])
        df = df[[projection, *BASIC_ATTR_DICT.keys()]]
        st.markdown("""
                    ### Box plot
                """)
        fig = px.box(df, y=projection)
        st.plotly_chart(fig)

        st.markdown("""
            ### Histogram
        """)
        for key in BASIC_ATTR_DICT.keys():
            if key != hue:
                filter_key = key
                break
        if len(hue) == 0:
            st.warning("Take into account that there is no hue selected")
            fig = px.histogram(df, x=projection)
        else:
            fig = px.histogram(df, x=projection, color=hue)
        st.plotly_chart(fig, theme="streamlit")


def get_filter(df, selected_property, container, hue):
    df = df.groupby(by=['band', hue]).mean().reset_index()
    df.sort_values(by='band')
    fig, ax = plt.subplots(1, 1)
    ax.set_title(selected_property + " by mean")
    sns.barplot(df, x='band', y='value', hue=hue, ax=ax)
    container.write(fig)


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

    comparing_distribution(df)
    comparing_boxplot(df)

    comparing_histograms(df, selected_bands)


def comparing_histograms(df, selected_bands):
    st.markdown(
        """
        ### Histograms
        """
    )
    show_histograms = st.checkbox("Show histograms")
    if show_histograms:
        for b in selected_bands:
            fig = px.histogram(df[df.band == int(b)],
                               title=f"Histogram of {select_property} in band {b}",
                               x="value", color=hue if hue != "" else None)
            st.plotly_chart(fig, theme="streamlit", )


def comparing_boxplot(df):
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


def comparing_distribution(df):
    st.markdown(
        f"""
        ### Mean of {select_property}
        """
    )
    if len(hue) == 0:
        st.warning("Please, select a hue from the sidebar")
        return
    get_filter(df, select_property, st, hue)


def comparing_properties(df):
    comparing_container = st.container()
    with comparing_container:
        remaining_properties = set(NAME_PROPERTIES)
        remaining_properties.remove(select_property)
        compared_property = st.selectbox("Select a property to compare", remaining_properties)
        select_all_bands = st.checkbox("Select all bands", key=43)

        if select_all_bands:
            selected_bands = st.multiselect("Select bands to display", BANDS, BANDS, disabled=True)
        else:
            selected_bands = st.multiselect("Select bands to display", BANDS)
        if len(selected_bands) == 0:
            st.warning("Please, select a band to continue.")
            return
        projection = [
                "{band}_{property}".format(band=band, property=PROPERTIES_DICT[property]) for band in selected_bands for property in [select_property, compared_property]
            ]

        marker_size = st.number_input("Size of the markers", 0.1, 4.0, step=0.1, value=3.0, )
        samples = st.slider("Select % of samples", 1, 100)
        samples_int = int(df.shape[0] * samples / 100)
        st.markdown(
            f"The sample is composed by {samples_int} rows ({samples_int // 13} patches) from the former {df.shape[0]} rows ({df.shape[0] // 13} patches)")
        df = df.sample(samples_int)
        df = get_full_df_projection(df, projection)
        for band in selected_bands:
            band = int(band)
            new_df = pd.DataFrame()
            name_band_property = f"{band}_{PROPERTIES_DICT[select_property]}"
            new_df[select_property] = df[df.measure == name_band_property]['value']
            name_band_property = f"{band}_{PROPERTIES_DICT[compared_property]}"
            new_df[compared_property] = df[df.measure == name_band_property]['value']
            new_df[hue] = df[df.measure == name_band_property][hue]
            if hue == "":
                fig = px.scatter(new_df, x=select_property, y=compared_property,
                                 title="Scatter plot of {} and {} in band {}".format(select_property, compared_property,
                                                                                     band))
            else:
                fig = px.scatter(new_df, x=select_property, y=compared_property, title="Scatter plot of {} and {} in band {}".format(select_property, compared_property, band), color=hue)
            fig.update_traces(marker_size=marker_size)
            fig.update_yaxes(matches="y")
            fig.update_xaxes(matches="x")
            st.plotly_chart(fig)

concatenated = pd.concat([cloudless, cloudy])

select_property = st.sidebar.selectbox("Select a property", NAME_PROPERTIES)
hue = st.sidebar.selectbox("Select a hue", [""] + list(BASIC_ATTR_DICT.keys()))
possible_filters = set(BASIC_ATTR_DICT.keys())
filter_key = st.sidebar.selectbox("Please, select a filter", [""] + list(possible_filters))
if filter_key != "":
    filters = st.sidebar.multiselect("Please, filter by these options", BASIC_ATTR_DICT[filter_key])
    if len(filters) != 0:
        last_shape = concatenated.shape[0]
        concatenated = concatenated[concatenated[filter_key].apply(lambda x: x in list(map(str, filters)))]
        st.sidebar.write("By filtering, it is showing {} rows of {} ({:.2f} %)".format(concatenated.shape[0], last_shape,
                                                                                       concatenated.shape[
                                                                                           0] / last_shape * 100))
    else:
        st.sidebar.warning("Any filter selected.")
tab1, tab2, tab3 = st.tabs(['Analysis of a band', 'Comparison between bands', 'Comparison between properties'])

with tab1:
    st.markdown("""
        ## Analysing a band
    """)
    band_analysis(concatenated)

with tab2:
    st.markdown(f"""
        ## Comparing bands regarding a property
        Compare between two or more bands regarding the property specified. In this case, *{select_property}*.
    """)
    comparing_bands(concatenated)

with tab3:
    st.markdown("""
        ## Comparing properties
    """)
    comparing_properties(concatenated)
