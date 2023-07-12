# page2.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots


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

seasons = ['spring', 'summer', 'fall', 'winter']


def add_title(text, level='#'):
    st.markdown(
        f"""
        {level} {text}
        """
    )


add_title("Cloud coverage")


def cloudless_body():
    add_title("Cloudless", "###")
    fig = px.histogram(cloudless,
                       title="Cloud coverage distribution in cloudless images",
                       x="cloud_percentage")
    st.plotly_chart(fig, theme="streamlit", )


def cloudy_body():
    add_title("Cloudy", "###")
    fig = px.histogram(cloudy,
                       title="Cloud coverage distribution in cloudy images",
                       x="cloud_percentage")
    st.plotly_chart(fig, theme="streamlit", )
    fig = make_subplots(rows=2, cols=2, subplot_titles=seasons)
    idxs = [(x, y) for x in [1, 2] for y in [1, 2]]
    for (row, col), season in zip(idxs, seasons):
        data = px.histogram(cloudy[cloudy.season == season],
                            title="Cloud coverage in cloudy images",
                            x="cloud_percentage", ).data[0]
        fig.add_trace(data, row=row, col=col)
    st.plotly_chart(fig, theme="streamlit", )


def seasons_body():
    type_options = st.multiselect("Select type", ["Cloudless", "Cloudy"], ["Cloudless", "Cloudy"])
    if len(type_options) == 0:
        st.warning("You have to select the type to populate the dataset")
        return
    elif len(type_options) == 2:
        df = pd.concat([cloudless, cloudy])
    elif type_options[0] == "Cloudless":
        df = cloudless
    else:
        df = cloudy
    fig = px.histogram(df,
                       title="Cloud coverage distribution",
                       x="cloud_percentage", color="season")
    st.plotly_chart(fig, theme="streamlit", )


def all_body():
    # add_title("All", "###")
    cloudless['type'] = 'Cloudless'
    cloudy['type'] = 'Cloudy'
    fig = px.histogram(pd.concat([cloudless, cloudy]),
                       title="Cloud coverage distribution",
                       x="cloud_percentage", color="type")
    st.plotly_chart(fig, theme="streamlit", )


def get_crops_attr(bands):
    return [f"b{b}_cloudy_crop_correlation_mse" for b in bands]


def get_bands_attr(bands):
    return [f"b{b}_cloudy_band_correlation_mse" for b in bands]


def correlation_body():
    add_title("Correlation between bands and cloud mask", "###")
    BANDS = list(range(13))
    df = cloudy.copy().head(100)
    col1, col2, col3 = st.columns(3)
    with col1:
        option = st.selectbox("Select the RoI", ('Cropped', 'All Image', 'Both'))
    with col3:
        st.write("Other options:")
        selected = st.checkbox("Select all")
    with col2:
        if selected:
            selected_bands = BANDS
        else:
            selected_bands = st.multiselect("Select bands", BANDS)
    if option == "Cropped":
        crops = get_crops_attr(selected_bands)
        season = st.selectbox("Select Season", ["All"] + seasons)
        if season == "All":
            correlation_crop(df, crops)
        else:
            correlation_crop(df[df.season == season], crops)
    elif option == "All Image":
        select_bands = get_bands_attr(selected_bands)
        season = st.selectbox("Select Season", ["All"] + seasons)
        if season == "All":
            correlation_band(df, select_bands)
        else:
            correlation_band(df[df.season == season], select_bands)
    else:
        crops = get_crops_attr(selected_bands)
        select_bands = get_bands_attr(selected_bands)

        season = st.selectbox("Select Season", ["All"] + seasons)
        df = df if season == "All" else df[df.season == season]
        correlation_crop(crops, df)
        correlation_band(df, select_bands)
    add_title("Relation between cropped and all mask", "####")
    correlation_between(df, selected_bands)


def correlation_between(df, selected_bands):
    crops = get_crops_attr(selected_bands)
    select_bands = get_bands_attr(selected_bands)
    df = df[[*crops, *select_bands, 'cloud_percentage']]
    df = df[~(df == -1).any(axis=1)]

    df[crops] =  (df[crops].apply(lambda x: x * df['cloud_percentage']))
    st.latex(r"""
        \text{crop mse} \cdot \text{cloud percentage} - \text{mse} 
        """)
    df[select_bands] = df[select_bands].apply(lambda x: x * (1 - df['cloud_percentage']))

    df[selected_bands] = (np.array(df[crops]) + np.array(df[select_bands]))

    st.write(df[selected_bands].mean())


def correlation_band(df, select_bands):
    add_title("Mask correlation of all the image", "####")
    if len(select_bands) == 0:
        st.warning("Please select at least one band.")
        return
    df = df.transpose().loc[select_bands]
    indexes = df.reset_index()['index']
    bs = indexes.apply(lambda x: x.split('_')[0])
    df = df.transpose()
    df = df.rename(columns=dict(zip(indexes, bs)))
    fig = px.box(df)
    st.plotly_chart(fig, theme="streamlit")


def correlation_crop(df, select_bands):
    add_title("Mask correlation of the image crop mask", "####")
    if len(select_bands) == 0:
        st.warning("Please select at least one band.")
        return
    df['is_na'] = df[select_bands[0]] == -1
    st.markdown("""
    There is a total of {} non-covered images over {} ({:.2f} %).

    For that reason, {} will be taken into account to avoid zero-division.
    """.format(df['is_na'].sum(),
               df.count()['is_na'],
               df['is_na'].sum() /
               df.count()['is_na'] * 100, (~df['is_na']).sum()), )
    fig = px.bar(df, x='is_na')
    st.plotly_chart(fig, theme="streamlit")
    df = df[~df['is_na']]
    df = df.transpose().loc[select_bands]
    indexes = df.reset_index()['index']
    bs = indexes.apply(lambda x: x.split('_')[0])
    df = df.transpose()
    df = df.rename(columns=dict(zip(indexes, bs)))
    fig = px.box(df)
    st.plotly_chart(fig, theme="streamlit")


# cloudless_body()
# cloudy_body()
distribution_tab, correlation_tab = st.tabs(['Distribution of cloud %', 'Band cloud correlation'])
with distribution_tab:
    add_title("Cloud coverage distributions", "##")
    all_body()
    seasons_body()
with correlation_tab:
    correlation_body()

# TODO: MASK
