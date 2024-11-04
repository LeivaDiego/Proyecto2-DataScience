import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set page configuration
st.set_page_config(page_title="Analisis de Vértebras Cervicales", layout="centered")

# Title of the app
st.title("Analisis de Vértebras Cervicales")

# Section to upload a zip (mockup)
st.header("Carga un Folder o Zip con las imágenes de la tomografía")
st.file_uploader("Escoge un ZIP file", type=["zip"])

# Placeholder for data analysis section
st.header("Visualización de Diagnóstico")

# Display a mock data preview table
st.write("Resultados:")

# Mock plot
data = {
    'vertebra': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
    'fractured': [100, 300, 50, 75, 60, 200, 150],
    'not_fractured': [1900, 1700, 1950, 1925, 1940, 1800, 1850]
}
df = pd.DataFrame(data)

# Plotting a stacked bar chart
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(df['vertebra'], df['not_fractured'], label='not_fractured', color='yellowgreen')
ax.bar(df['vertebra'], df['fractured'], bottom=df['not_fractured'], label='fractured', color='indianred')

# Customize the plot
ax.set_title("Fractures as per Vertebra")
ax.set_xlabel("Vertebra")
ax.set_ylabel("Value")
ax.legend(title="Variable")
st.pyplot(fig)


# Mockup GIF display
st.header("Animación de la tomografía")
st.image("https://media.tenor.com/JhHYrYgP_qgAAAAM/abdominal-ct-scan.gif", caption="Tomografía")

# Mock text description
st.write("Aproximadamente 2 vértebras fracturadas.")