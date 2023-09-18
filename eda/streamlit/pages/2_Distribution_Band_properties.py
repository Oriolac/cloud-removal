import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


class KeyContainer:

    def __init__(self):
        self.key = 0

    def new(self):
        to = self.key
        self.key += 1
        return to


BANDS = [f"{b}" for b in range(13)]


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

keys = KeyContainer()

PROPERTIES_DICT = {
    "Mean": "mean",
    "Median": "median",
    "Percentile 25": "p25",
    "Percentile 75": "p75",
    "std": "std",
}

BASIC_ATTR_DICT = {
    "season": ["spring", "summer", "fall", "winter"],
    "type": ["cloudless", "cloudy"],
}

PALETTES = {
    "season": {
        'spring': (90 / 255, 213 / 255, 127 / 255),
        'summer': (233 / 255, 204 / 255, 87 / 255),
        'fall': (182 / 255, 114 / 255, 80 / 255),
        'winter': (80 / 255, 158 / 255, 182 / 255),
    },
    "type": {
        'cloudless': (0.2, 0.6, 0.8),
        'cloudy': (0.9, 0.5, 0.4),
    }
}

SEASON_ORDER = {
    "spring": 0,
    "summer": 1,
    "fall": 2,
    "winter": 3
}

NAME_PROPERTIES = list(PROPERTIES_DICT.keys())


def get_full_df_projection(old_df, projection):
    properties = [f"{b}_{prop}" for b in range(13) for prop in projection]
    df = pd.DataFrame()
    for attr in properties:
        df1 = pd.DataFrame()
        df1['value'] = old_df[attr]
        df1['measure'] = attr.split('_')[1]
        df1['band'] = int(attr.split('_')[0])
        for copy_attr in BASIC_ATTR_DICT.keys():
            df1[copy_attr] = old_df[copy_attr]
        df = pd.concat([df, df1])
    return df


def filter_season(xs):
    return SEASON_ORDER[xs]


def get_filter(df, select_bands, selected_property, hue):
    projection = [
        "{band}_{property}".format(band=band, property=PROPERTIES_DICT[selected_property]) for band in select_bands
    ]
    df = df[df.measure == PROPERTIES_DICT[selected_property]]
    df = df[df.band.apply(lambda x: str(x) in select_bands)]
    df = df[[hue, 'band', 'value']].groupby(by=['band', hue]).mean().reset_index()
    df.sort_values(by='band')
    fig, ax = plt.subplots(1, 1)
    ax.set_title(selected_property)
    if hue == "season":
        df = df.sort_values(by=hue, key=np.vectorize(filter_season))
        sns.barplot(df, x='band', y='value', hue=hue, ax=ax, palette=PALETTES[hue])
    else:
        sns.barplot(df, x='band', y='value', hue=hue, ax=ax, palette=PALETTES[hue])
    st.write(fig)


def to_kind_hist(df, bands, hue, width=0.3, spacing=0.3 ):
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    for i_band, band in enumerate(bands):
        band = int(band)
        df_band = df[df.band == band]
        df_band = df_band[['measure', hue, 'value']].groupby(by=['measure', hue]).mean()[['value']]
        ax.grid(zorder=-1)
        x_start = width * 2 + i_band * spacing + len(BASIC_ATTR_DICT[hue]) * i_band * width
        rects = []
        for i, hue_value in enumerate(BASIC_ATTR_DICT[hue]):
            rect = Rectangle((x_start + width * i, df_band.loc['p25', hue_value]['value']), width,
                                 df_band.loc['p75', hue_value]['value'], color=PALETTES[hue][hue_value], lw=2, zorder=2)
            ax.add_patch(rect, )
            ax.plot([x_start + width * i, x_start + (width * (i+1))], [df_band.loc['median', hue_value]] * 2, color=(1, 1, 1), zorder=3)
            mean, = ax.plot((x_start + width / 2 + width * i), df_band.loc['mean', hue_value], marker='o', color=(0, 0, 0), zorder=3,
                    markersize=5)
            if i_band == 0 and i == 0:
                mean.set_label("Mean value")
            rects.append((rect, hue_value))
        if i_band == 0:
            for rect, value in rects:
                rect.set_label(value)
        ax.text(x_start + width * len(BASIC_ATTR_DICT[hue]) / 2, -10, f"Band: {band}", ha='center')
    ax.legend(loc="lower right")
    ax.set_title("Distribution properties")
    ax.get_xaxis().set_visible(False)
    plt.xlim([width, (len(bands)) * (width) * len(BASIC_ATTR_DICT[hue]) + (len(bands) - 1) * spacing + width * 3])
    plt.ylim([0, 255])
    st.write(fig)
    return df


bands_container = st.sidebar.container()
select_all_bands = st.sidebar.checkbox("Select all bands")
if select_all_bands:
    selected_bands = bands_container.multiselect("Select bands", BANDS, BANDS, disabled=True)
else:
    selected_bands = bands_container.multiselect("Select bands", BANDS)

hue = st.sidebar.selectbox("Please, select a hue", list(BASIC_ATTR_DICT.keys()))

st.markdown("""
    # Distribution of bandwidths
""")
df = get_full_df_projection(pd.concat([cloudless, cloudy]), ['mean', 'median', 'p25', 'p75', 'std'])
if len(selected_bands) == 0:
    st.warning("Please, select one band at least")
else:
    to_kind_hist(df, selected_bands, hue)
st.markdown("""
    ### Check one specific property
""")
select_properties = st.selectbox("Select properties", NAME_PROPERTIES, key=keys.new())
if len(selected_bands) == 0:
    st.warning("Please, select one band at least")
else:
    get_filter(df, selected_bands, select_properties, hue)
