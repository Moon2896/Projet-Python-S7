import streamlit as st
import pandas as pd

st.write(
    """
    # My first NAZI app
    HELLO COMRADS 
    """
)

df = pd.read_csv("spambase_prep_norm.csv", sep=";")
st.line_chart(df)