import streamlit as st
 
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
 
#st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")
st.sidebar.success("https://cheat-sheet.streamlit.app/")

 
st.write("""
# :rainbow[Benvenuti!] 🎈
Hello *world!*
""")

st.image("evision.png")