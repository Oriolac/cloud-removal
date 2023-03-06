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

bands_container = st.sidebar.container()
select_all_bands = st.sidebar.checkbox("Select all bands")

if select_all_bands:
    selected_bands = bands_container.multiselect("Select bands", BANDS, BANDS, disabled=True)
else:
    selected_bands = bands_container.multiselect("Select bands", BANDS)

PROPERTIES_DICT = {
    "Mean": "mean",
    "Median": "median",
    "Percentile 25": "p25",
    "Percentile 75": "p75",
    "std": "std",
}


NAME_PROPERTIES = list(PROPERTIES_DICT.keys())



def turn_to_dist(cloudless, cloudy):
    properties = [f"{b}_{prop}" for b in range(13) for prop in ["mean", "median", "p25", "p75"]]
    df = pd.DataFrame()
    for attr in properties:
        df1 = pd.DataFrame()
        df1['value'] = cloudless[attr]
        df1['band'] = int(attr.split('_')[0])
        df1['measure'] = attr.split('_')[1]
        df1['type'] = 'cloudless'
        df2 = pd.DataFrame()
        df2['value'] = cloudy[attr]
        df2['band'] = int(attr.split('_')[0])
        df2['measure'] = attr.split('_')[1]
        df2['type'] = 'cloudy'
        union = pd.concat([df1, df2])
        df = pd.concat([df, union])
    return df

def get_filter(df, select_bands, selected_property):
    projection = [
        "{band}_{property}".format(band=band, property=PROPERTIES_DICT[selected_property]) for band in select_bands
    ]
    df = df[df.measure == PROPERTIES_DICT[selected_property]]
    df = df[df.band.apply(lambda x: str(x) in select_bands)]
    df = df.groupby(by=['band', 'type']).mean().reset_index()
    df.sort_values(by='band')
    fig, ax = plt.subplots(1, 1)
    ax.set_title(selected_property)
    sns.barplot(df, x='band', y='value', hue='type', ax=ax)
    st.write(fig)




def to_kind_hist(df, bands, width=0.3, spacing=0.5):
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    for band in bands:
        band = int(band)
        df_band = df[df.band == band]
        df_band = df_band.groupby(by=['measure', 'type']).mean()[['value']]
        ax.grid(zorder=-1)
        x_start = (band + 1) * 2 * width + spacing * band
        print(x_start, x_start + width)
        rectless = Rectangle((x_start, df_band.loc['p25', 'cloudless']['value']), width,
                             df_band.loc['p75', 'cloudless']['value'], color=(0.2, 0.6, 0.8), lw=2, zorder=2)
        recty = Rectangle((x_start + width, df_band.loc['p25', 'cloudy']['value']), width,
                          df_band.loc['p75', 'cloudy']['value'], color=(0.9, 0.5, 0.4), lw=2, zorder=2)
        ax.plot([x_start, x_start + width], [df_band.loc['median', 'cloudless']] * 2, color=(1, 1, 1), zorder=3)
        ax.plot([x_start + width, x_start + width * 2], [df_band.loc['median', 'cloudy']] * 2, color=(1, 1, 1),
                zorder=3)
        ax.plot((x_start + width / 2), df_band.loc['mean', 'cloudless'], 'bo', zorder=3, markersize=5)
        ax.plot((x_start + width / 2 + width), df_band.loc['mean', 'cloudy'], '-ro', zorder=3, markersize=5)
        ax.text(x_start + width, -10, f"Band: {band}")
        ax.add_patch(rectless, )
        ax.add_patch(recty, )

    ax.set_title("Distribution properties")
    ax.get_xaxis().set_visible(False)
    plt.xlim([width, (len(bands) + 1) * 2 * width + spacing * (len(bands)) - (spacing) + width])
    plt.ylim([0, 255])
    st.write(fig)
    return df


st.markdown("""
    # Distribution of bandwidths
""")
DF = turn_to_dist(cloudless, cloudy)

if len(selected_bands) == 0:
    st.warning("Please, select one band at least")
else:
    to_kind_hist(DF, selected_bands)
st.markdown("""
    ### Check one specific property
""")
select_properties = st.selectbox("Select properties", NAME_PROPERTIES, key=keys.new())
if len(selected_bands) == 0:
    st.warning("Please, select one band at least")
else:
    get_filter(DF, selected_bands, select_properties)
