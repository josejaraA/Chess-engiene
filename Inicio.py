import streamlit as st
from PIL import Image

st.set_page_config(page_title="App de Entrenamiento y Juego de Ajedrez", layout="wide")


st.title("Bienvenido a la App de Ajedrez y Entrenamiento")

col1, col2, col3 = st.columns([1, 4, 1])  

with col2:  
    image = Image.open("imgn/img.png")  
    st.image(image, width=600) 


st.markdown("---")

with st.expander("📌 Instrucciones", expanded=True):
    st.write(
        """
        - **Análisis y Resultados:** Visualiza datos y gráficos de entrenamiento de distintos modelos y su respectivo análisis.
        - **Juego de Ajedrez:** Juega contra el modelo de ajedrez. Ingresa tu movimiento en formato UCI (por ejemplo, `e2e4`) y el sistema responderá con un movimiento.
        """
    )


st.write("")
st.write("🎯 ¡Explora las pestañas para comenzar!")




 