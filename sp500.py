import pandas as pd
import streamlit as st

@st.cache_data(ttl=86400)
def load_sp500():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={"Name": "Security"})
    return df[["Symbol", "Security"]]
