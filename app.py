# app.py
import streamlit as st

# Título de la aplicación
st.title('Mi Aplicación de Streamlit en Jupyter')

# Algún contenido
st.write("Esta es una simple aplicación de Streamlit ejecutándose dentro de un Jupyter Notebook.")

# Entrada de usuario
user_input = st.text_input("Introduce algún texto")

# Mostrar la entrada
st.write("Has introducido:", user_input)