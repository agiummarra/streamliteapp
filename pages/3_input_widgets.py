import streamlit as st
import pandas as pd

st.markdown("# Input widgets 🧩")
st.sidebar.markdown("# Input widgets 🧩")

st.subheader('Slider', divider='rainbow')
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

st.subheader('Camera', divider='rainbow')
x = st.camera_input("Camera input", help="Questo è l'help del componente camera_input.")

if x is not None:
    st.image(x)

#y = st.camera_input("Label2", help="help2", disabled=True)