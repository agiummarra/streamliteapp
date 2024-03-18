import streamlit as st
from annotated_text import annotated_text

st.markdown("# Elementi testo 💬")
st.header('Header', divider='grey')
st.sidebar.markdown("# Elementi testo 💬")

st.subheader('Questo è un subheader con un divider rainbow', divider='rainbow')
st.subheader('_Streamlit_ is :blue[cool] :sunglasses:')

st.divider()

st.subheader('Annotated Text', divider='rainbow')

annotated_text(
    "This ",
    ("is", "Verb", "#8ef"),
    " some ",
    ("annotated", "Adj", "#faa"),
    ("text", "Noun", "#afa"),
    " for those of ",
    ("you", "Pronoun", "#fea"),
    " who ",
    ("like", "Verb", "#8ef"),
    " this sort of ",
    ("thing", "Noun", "#afa"),
    ". "
    "And here's a ",
    ("word", "", "#faf"),
    " with a fancy background but no label.",
)

