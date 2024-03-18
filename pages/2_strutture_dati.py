import streamlit as st
import pandas as pd

st.markdown("# Strutture dati 🎉")
st.sidebar.markdown("# Strutture dati 🎉")

st.subheader('Tabella creata a mano', divider='rainbow')
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

st.subheader('Tabella dei :rainbow[moduli] di :red[CDA]:blue[Plus]', divider='rainbow')
st.write("""
**Dettagli:**
> - tipo: **:green[mysql]**
> - database: **:green[cdap]**
> - tabella: **:green[module]**
""") 
st.code("select * from module")

conn = st.connection("cdap")
df = conn.query("select * from module")
st.dataframe(df)

st.table(df.iloc[0:10])

st.subheader('Json', divider='rainbow')
st.json({'foo':'bar','fu':'ba'})

st.subheader('Metriche', divider='rainbow')
st.metric(label="Prezzo Acqua Sant'anna Naturale Lt. 1,5 X 6 Bottiglie", value="3.42 €", delta="- 0.20 €", delta_color="inverse")

col1, col2, col3 = st.columns(3)
col1.metric("Ultimo mese", "3.50 €", "+ 0.50 €", delta_color="inverse")
col2.metric("Ultimo semestre", "3.35 €", "- 0.35 €", delta_color="inverse")
col3.metric("Ultimo anno", "3.40 €", " + 0.45 €", delta_color="inverse")

#st.metric(label="Gas price", value=4, delta=-0.5, delta_color="inverse")
#st.metric(label="Active developers", value=123, delta=123, delta_color="off")

#st.divider()

#st.write("""
### Tabella mysql "firebase_sviluppo.fbconnection"
#""")
#conn = st.connection("firebase_sviluppo")
#df = conn.query("select * from fbconnection")
#st.dataframe(df)
