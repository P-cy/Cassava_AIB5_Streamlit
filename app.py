import streamlit as st
import numpy as np
import pandas as pd

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [13.76514, 100.53829],
    columns=['lat', 'lon'])

st.map(map_data)
