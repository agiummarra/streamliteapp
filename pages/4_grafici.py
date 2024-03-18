import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from geopy.geocoders import Nominatim

st.markdown("# Grafici 📊")
st.sidebar.markdown("# Grafici 📊")

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data) 

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

#df = pd.read_csv("/Users/andrea/vscode-workspace/python/streamlit/pages/Interventi.csv")
#st.line_chart(df)

chart_data = pd.DataFrame(
   np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
   columns=['lat', 'lon'])

# calling the Nominatim tool and create Nominatim class
##loc = Nominatim(user_agent="MyLibraryApp")

# entering the location name
##location = loc.geocode("VIALE EUROPA N. 65  - 97100 RAGUSA (RG)")

# printing address
##print(location.address)

# printing latitude and longitude
##print("Latitude = ", location.latitude)
##print("Longitude = ", location.longitude)

#chart_data = pd.DataFrame(
#    np.array([[1,36.92], [1,14.71]]),
#    columns=['lat', 'lon'])

#print("chart_data: ", chart_data)

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=37.76,
        longitude=-122.4,
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=chart_data,
           get_position='[lon, lat]',
           radius=200,
           elevation_scale=4,
           elevation_range=[0, 1000],
           pickable=True,
           extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=chart_data,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
        ),
    ],
))