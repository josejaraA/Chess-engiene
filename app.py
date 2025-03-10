import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import chess
import chess.svg
import streamlit.components.v1 as components
import random
import numpy as np

# =============================
# Funciones para procesamiento y gráficos
# =============================


def convert_percentage(df):
    df['Accuracy (Train)'] = df['Accuracy (Train)'].str.rstrip('%').astype(float) / 100
    df['Top-5 Accuracy (Train)'] = df['Top-5 Accuracy (Train)'].str.rstrip('%').astype(float) / 100
    df['Accuracy (Val)'] = df['Accuracy (Val)'].str.rstrip('%').astype(float) / 100
    df['Top-5 Accuracy (Val)'] = df['Top-5 Accuracy (Val)'].str.rstrip('%').astype(float) / 100
    return df

def convert_percentage_(df):
    df['From Square Accuracy (Train)'] = df['From Square Accuracy (Train)'].str.rstrip('%').astype(float) / 100
    df['To Square Accuracy (Train)'] = df['To Square Accuracy (Train)'].str.rstrip('%').astype(float) / 100
    df['From Square Accuracy (Val)'] = df['From Square Accuracy (Val)'].str.rstrip('%').astype(float) / 100
    df['To Square Accuracy (Val)'] = df['To Square Accuracy (Val)'].str.rstrip('%').astype(float) / 100
    return df

def plot_mosaic(df, title_prefix):
    st.subheader(f"{title_prefix} - Gráficos de Evolución")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    def plot_metric(ax, train_col, val_col, title):
        ax.plot(df['Epoch'], df[train_col], label='Train', marker='o')
        ax.plot(df['Epoch'], df[val_col], label='Validation', marker='s')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
    
    plot_metric(axes[0, 0], 'Accuracy (Train)', 'Accuracy (Val)', "Accuracy")
    plot_metric(axes[0, 1], 'Loss (Train)', 'Loss (Val)', "Loss")
    plot_metric(axes[1, 0], 'Top-5 Accuracy (Train)', 'Top-5 Accuracy (Val)', "Top-5 Accuracy")
    
    axes[1, 1].plot(df['Epoch'], df['Learning Rate'], label='Learning Rate', marker='d', color='purple')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].legend()
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_mosaico_(df, title_prefix):
    st.subheader(f"{title_prefix} - Gráficos de Evolución")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    def plot_metric(ax, train_col, val_col, title):
        ax.plot(df['Epoch'], df[train_col], label='Train', marker='o')
        ax.plot(df['Epoch'], df[val_col], label='Validation', marker='s')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
    
    plot_metric(axes[0, 0], 'From Square Accuracy (Train)', 'From Square Accuracy (Val)', "Accuracy - pieza")
    plot_metric(axes[0, 1], 'Loss (Train)', 'Loss (Val)', "Loss")
    plot_metric(axes[1, 0], 'To Square Accuracy (Train)', 'To Square Accuracy (Val)', "Accuracy - movimiento")
    
    axes[1, 1].plot(df['Epoch'], df['Learning Rate'], label='Learning Rate', marker='d', color='purple')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].legend()
    
    plt.tight_layout()
    st.pyplot(fig)

# =============================
# Estructura 
# =============================

st.set_page_config(page_title="App de Entrenamiento y Juego de Ajedrez", layout="wide")
tabs = st.tabs(["Inicio", "Resultados", "Juego de Ajedrez"])

# -----------------------------
# Pestaña: Inicio
# -----------------------------
with tabs[0]:
    st.title("Bienvenido a la App de Ajedrez y Entrenamiento")
    st.write("""
    **Instrucciones:**
    - En la pestaña **Resultados** podrás visualizar los datos y gráficos de entrenamiento de distintos modelos.
    - En la pestaña **Juego de Ajedrez** podrás jugar contra el modelo de ajedrez.  
      Ingresa tu movimiento en formato UCI (por ejemplo, `e2e4`) y el sistema responderá con un movimiento.
    """)

# -----------------------------
# Pestaña: Resultados
# -----------------------------
with tabs[1]:
    st.title("Resultados de Entrenamiento del Modelo")
    
    df1 = pd.read_csv("Resultados/training_results_modelo1_1.csv", index_col="Epoch")
    df1_2 = pd.read_csv("Resultados/training_results_modelo1_2.csv", index_col="Epoch")
    df2 = pd.read_csv('Resultados/training_results_modelo2_1.csv', index_col='Epoch')
    df3 = pd.read_csv('Resultados/training_results_modelo3_1.csv', index_col='Epoch')
    df3_2 = pd.read_csv('Resultados/training_results_modelo3_2.csv', index_col='Epoch')
    df3_3 = pd.read_csv('Resultados/training_results_modelo3_3.csv', index_col='Epoch')
    df4 = pd.read_csv('Resultados/training_results_modelo4_1.csv', index_col='Epoch')
    df4_2 = pd.read_csv('Resultados/training_results_modelo4_2.csv', index_col='Epoch')



    dfs = [df1, df1_2, df2, df3, df3_2, df3_3, df4, df4_2]

    for df in dfs:
        df.reset_index(inplace=True)
    
    df1 = convert_percentage(df1)
    df1_2 = convert_percentage(df1_2)
    df2 = convert_percentage(df2)
    df3 = convert_percentage(df3)
    df3_2 = convert_percentage(df3_2)
    df3_3 = convert_percentage(df3_3)
    df4 = convert_percentage_(df4)
    df4_2 = convert_percentage_(df4_2)



    
    st.subheader("Modelo 1.1 - Datos de Entrenamiento - Learning Rate 1")
    st.dataframe(df1.set_index('Epoch'))
    plot_mosaic(df1, "Modelo 1.1 - LR1")
    
    st.subheader("Modelo 1.2 - Datos de Entrenamiento - LR 2")
    st.dataframe(df1_2.set_index('Epoch'))
    plot_mosaic(df1_2, "Modelo 1.2")
    
    st.subheader("Modelo 2 - Datos de Entrenamiento")
    st.dataframe(df2.set_index('Epoch'))
    plot_mosaic(df2, "Modelo 2")
    
    st.subheader("Modelo 3.1 - Datos de Entrenamiento - LR 1")
    st.dataframe(df3.set_index('Epoch'))
    plot_mosaic(df3, "Modelo 3.1")
    
    st.subheader("Modelo 3.2 - Datos de Entrenamiento - LR 2")
    st.dataframe(df3_2.set_index('Epoch'))
    plot_mosaic(df3_2, "Modelo 3.2")
    
    st.subheader("Modelo 3.3 - Datos de Entrenamiento - LR 3")
    st.dataframe(df3_3.set_index('Epoch'))
    plot_mosaic(df3_3, "Modelo 3.3")
    
    st.subheader("Modelo 4.1 - Datos de Entrenamiento - LR 1")
    st.dataframe(df4.set_index('Epoch'))
    plot_mosaico_(df4, "Modelo 4.1")
    
    st.subheader("Modelo 4.2 - Datos de Entrenamiento - LR 2")
    st.dataframe(df4_2.set_index('Epoch'))
    plot_mosaico_(df4_2, "Modelo 4.2")


# -----------------------------
# Pestaña: Juego de Ajedrez
# -----------------------------
with tabs[2]:
    st.title("Juego de Ajedrez con el Modelo")
    custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
    model = tf.keras.models.load_model("ModeloAjedrez_v8.h5", custom_objects=custom_objects)
     




def board_to_input(board):
    fen = board.fen()
    input_vector = np.zeros((1, 773))  
    return input_vector


def decode_prediction(prediction, board):

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    if prediction.shape[1] == len(legal_moves):
        idx = np.argmax(prediction[0])
        best_move = legal_moves[idx]
    else:
        best_move = random.choice(legal_moves)
    return best_move

def obtener_mejor_jugada(board, model):
    input_data = board_to_input(board)
    prediction = model.predict(input_data)
    best_move = decode_prediction(prediction, board)
    return best_move

if 'board' not in st.session_state:
    st.session_state.board = chess.Board()

board_svg = chess.svg.board(st.session_state.board, size=350)
components.html(board_svg, height=400)

user_move = st.text_input("Ingresa tu movimiento (formato UCI, e.g. 'e2e4'):")

if st.button("Realizar movimiento"):
    try:
        move = chess.Move.from_uci(user_move.strip())
        if move in st.session_state.board.legal_moves:
            st.session_state.board.push(move)
        else:
            st.error("Movimiento ilegal. Por favor, ingresa un movimiento válido.")
    except Exception as e:
        st.error("Error al interpretar el movimiento. Asegúrate de usar el formato UCI.")

    
    board_svg = chess.svg.board(st.session_state.board, size=350)
    components.html(board_svg, height=400)

if st.button("Movimiento del Modelo"):
    if st.session_state.board.is_game_over():
        st.info("El juego ha terminado.")
    else:
        mejor_movimiento = obtener_mejor_jugada(st.session_state.board, model)
        if mejor_movimiento is not None:
            st.session_state.board.push(mejor_movimiento)
            st.success(f"El modelo juega: {mejor_movimiento.uci()}")
        else:
            st.error("No se pudo obtener un movimiento del modelo.")
    board_svg = chess.svg.board(st.session_state.board, size=350)
    components.html(board_svg, height=400)