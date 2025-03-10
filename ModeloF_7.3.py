import chess 
import chess.pgn
import numpy as np
import tensorflow as tf
import random

########################################
# 1. Función para parsear partidas PGN sin mezclar movimientos en train/test
########################################
def parse_pgn_positions(file_path):
    games = []
    with open(file_path, 'r') as f:
        game = chess.pgn.read_game(f)
        while game:
            positions = []
            board = game.board()
            for move in game.mainline_moves():
                if move in board.legal_moves:
                    fen = board.fen()
                    positions.append({"board": fen, "move": move.uci()})
                    board.push(move)
            games.append(positions)
            game = chess.pgn.read_game(f)
    return games

########################################
# 2. Representación del tablero en matriz (8x8x17)
########################################
def board_to_matrix(fen):
    board = chess.Board(fen)
    matrix = np.zeros((8, 8, 17), dtype=np.float32)
    
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        channel = piece.piece_type - 1
        if piece.color == chess.BLACK:
            channel += 6
        matrix[row, col, channel] = 1
    
    # Agregar información adicional
    matrix[:, :, 12] = 1 if board.turn == chess.WHITE else -1
    castling = [chess.BB_A1, chess.BB_H1, chess.BB_A8, chess.BB_H8]
    for i in range(4):
        matrix[:, :, 13 + i] = 1 if board.has_castling_rights([chess.WHITE, chess.WHITE, chess.BLACK, chess.BLACK][i]) else 0
    matrix[:, :, 16] = 1 if board.has_legal_en_passant() else 0
    
    return matrix

########################################
# 3. Codificación de movimientos: predicción en dos pasos
########################################
def encode_move(board, move_uci):
    move = chess.Move.from_uci(move_uci)
    if move in board.legal_moves:
        return move.from_square, move.to_square
    else:
        return None, None  # Filtramos jugadas ilegales

########################################
# 4. Preparación del dataset sin mezclar partidas
########################################
pgn_file = "datasets2.pgn"
games = parse_pgn_positions(pgn_file)

# Barajar partidas para evitar sesgo de orden
random.shuffle(games)

# Dividir por partidas (80% train, 20% test)
split_index = int(0.8 * len(games))
train_dataset = [pos for game in games[:split_index] for pos in game]
test_dataset = [pos for game in games[split_index:] for pos in game]

X_train = np.array([board_to_matrix(entry["board"]) for entry in train_dataset], dtype=np.float32)
y_train_from = np.array([encode_move(chess.Board(entry["board"]), entry["move"])[0] for entry in train_dataset], dtype=np.int32)
y_train_to = np.array([encode_move(chess.Board(entry["board"]), entry["move"])[1] for entry in train_dataset], dtype=np.int32)

X_test = np.array([board_to_matrix(entry["board"]) for entry in test_dataset], dtype=np.float32)
y_test_from = np.array([encode_move(chess.Board(entry["board"]), entry["move"])[0] for entry in test_dataset], dtype=np.int32)
y_test_to = np.array([encode_move(chess.Board(entry["board"]), entry["move"])[1] for entry in test_dataset], dtype=np.int32)


########################################
# 5. Definir la Arquitectura del Modelo
########################################
inputs = tf.keras.Input(shape=(8, 8, 17))
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)

def res_block(x, filters):
    shortcut = x
    y = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Conv2D(filters, (3, 3), activation=None, padding='same')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Add()([shortcut, y])
    y = tf.keras.layers.Activation('relu')(y)
    return y

x = res_block(x, 128)  # 🔹 Volvemos a una sola capa residual

x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.Dropout(0.35)(x)  # 🔹 Menos dropout para permitir mejor ajuste

output_from = tf.keras.layers.Dense(64, activation='softmax', name="from_square")(x)
output_to = tf.keras.layers.Dense(64, activation='softmax', name="to_square")(x)

model = tf.keras.Model(inputs=inputs, outputs=[output_from, output_to])

########################################
# 6. Compilación y Entrenamiento Ajustado
########################################
optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004)  # 🔹 Reducimos LR inicial

model.compile(
    optimizer=optimizer,
    loss={"from_square": "sparse_categorical_crossentropy", "to_square": "sparse_categorical_crossentropy"},
    metrics={"from_square": "accuracy", "to_square": "accuracy"}
)

history = model.fit(
    X_train, {"from_square": y_train_from, "to_square": y_train_to},
    epochs=50,
    batch_size=8,
    validation_data=(X_test, {"from_square": y_test_from, "to_square": y_test_to}),
    verbose=2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)  # 🔹 Más agresivo
    ]
)

ruta = 'ModeloAjedrez_v7.3.h5'
model.save(ruta)
print("Modelo guardado en:", ruta)