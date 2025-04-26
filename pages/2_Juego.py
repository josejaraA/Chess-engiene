import streamlit as st
import chess
import chess.svg
import tensorflow as tf
import numpy as np
import streamlit.components.v1 as components

# =============================
# Funciones para el Modelo De Ajedrez
# =============================

@st.cache_resource
def load_model():
    """Carga el modelo de ajedrez en caché para evitar recargas innecesarias."""
    custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}  # Mantener custom_objects
    return tf.keras.models.load_model("Modelos Finales\ModeloAjedrez_D2_4.keras", custom_objects=custom_objects)

def initialize_board():
    """Inicializa un nuevo tablero de ajedrez."""
    return chess.Board()

def render_board(board):
    """Renderiza el tablero de ajedrez en formato SVG."""
    return chess.svg.board(board=board, size=350)

def board_to_matrix(fen):
    """Convierte el tablero de ajedrez en una matriz que el modelo pueda procesar."""
    board = chess.Board(fen)
    matrix = np.zeros((8, 8, 17), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        channel = piece.piece_type - 1
        if piece.color == chess.BLACK:
            channel += 6
        matrix[row, col, channel] = 1

    matrix[:, :, 12] = 1 if board.turn == chess.WHITE else -1
    matrix[:, :, 13] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    matrix[:, :, 14] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    matrix[:, :, 15] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    matrix[:, :, 16] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0

    return matrix

def best_model_move(board, model):
    """Devuelve el mejor movimiento según el modelo."""
    mat = np.expand_dims(board_to_matrix(board.fen()), axis=0)  # (1,8,8,17)
    pred_from, pred_to, _ = model.predict(mat, verbose=0)

    best_prob = -1.0
    best_move = None
    for move in board.legal_moves:
        from_idx = move.from_square
        to_idx = move.to_square
        prob = pred_from[0][from_idx] * pred_to[0][to_idx]
        if prob > best_prob:
            best_prob = prob
            best_move = move

    return best_move

def mover_ia(model):
    """La IA hace su movimiento si la partida no ha terminado."""
    if not st.session_state.board.is_game_over():
        move = best_model_move(st.session_state.board, model)
        if move:
            st.session_state.board.push(move)
            st.success(f"La IA jugó: {move.uci()}")

# =============================
# Inicialización de Streamlit
# =============================
st.title("Juego de Ajedrez con el Modelo")

# Inicializar session_state
if "board" not in st.session_state:
    st.session_state.board = initialize_board()
if "model" not in st.session_state:
    st.session_state.model = load_model()
if "player_color" not in st.session_state:
    st.session_state.player_color = st.radio("Elige tu color:", ("Blancas", "Negras"))
    if st.session_state.player_color == "Negras":
        mover_ia(st.session_state.model)  # La IA mueve si el usuario juega con negras





# =============================
# Formulario para ingresar movimiento
# =============================
with st.form("move_form", clear_on_submit=True):
    user_move_uci = st.text_input("Ingresa tu movimiento (Ejemplo: e2e4):")
    submit_move = st.form_submit_button("Enviar Movimiento")

    if submit_move and user_move_uci.strip():
        try:
            move = chess.Move.from_uci(user_move_uci.strip())
            if move in st.session_state.board.legal_moves:
                st.session_state.board.push(move)
                st.success(f"Movimiento del usuario: {move.uci()}")

                # Si la partida no ha terminado, mueve la IA
                if not st.session_state.board.is_game_over():
                    mover_ia(st.session_state.model)
            else:
                st.error("Movimiento ilegal.")
        except ValueError:
            st.error("Formato inválido. Usa UCI, por ejemplo: e2e4.")

# =============================
# Mostrar nuevamente el tablero actualizado
# =============================
tablero_svg = render_board(st.session_state.board)
components.html(tablero_svg, height=400)

# =============================
# Verificar si la partida ha terminado
# =============================
if st.session_state.board.is_game_over():
    st.subheader("La partida ha finalizado.")
    st.write(f"Resultado: {st.session_state.board.result()}")

    if st.button("Reiniciar partida"):
        st.session_state.board = initialize_board()
        st.experimental_rerun()  # Recargar la aplicación para reiniciar el juego
