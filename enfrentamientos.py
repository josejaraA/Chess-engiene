import chess
import chess.pgn
import tensorflow as tf
import numpy as np
import random

def board_to_matrix(fen):
    board = chess.Board(fen)
    matrix = np.zeros((8, 8, 17), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        channel = piece.piece_type - 1
        if piece.color == chess.BLACK:
            channel += 6
        matrix[row, col, channel] = 1

    matrix[:, :, 12] = 1 if board.turn == chess.WHITE else -1

    derechos = [
        board.has_castling_rights(chess.WHITE),
        board.has_castling_rights(chess.WHITE),
        board.has_castling_rights(chess.BLACK),
        board.has_castling_rights(chess.BLACK),
    ]
    for i in range(4):
        matrix[:, :, 13 + i] = 1 if derechos[i] else 0

    matrix[:, :, 16] = 1 if board.has_legal_en_passant() else 0

    return matrix

def mejor_mov(board, model):
    matrix = board_to_matrix(board.fen())
    matrix = np.expand_dims(matrix, axis=0)
    pred_from, pred_to, _ = model.predict(matrix)
    pred_from = pred_from[0]
    pred_to = pred_to[0]

    mov_posibl = []
    for move in board.legal_moves:
        from_idx = move.from_square
        to_idx = move.to_square
        prob = pred_from[from_idx] * pred_to[to_idx]  
        mov_posibl.append((prob, move))

    mov_posibl.sort(reverse=True, key=lambda x: x[0])

    
    top_k = min(3, len(mov_posibl))
    selected_move = random.choices(mov_posibl[:top_k], weights=[m[0] for m in mov_posibl[:top_k]])[0][1]

    return selected_move

def jugar_partida(model_blancas, model_negras, nombre_blancas, nombre_negras):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Juego entre modelos"
    game.headers["White"] = nombre_blancas
    game.headers["Black"] = nombre_negras
    node = game

    ply_count = 0
    max_ply = 80 

    while not board.is_game_over() and ply_count < max_ply:
        if ply_count < 3:
            move = random.choice(list(board.legal_moves))
        else:
            if board.turn == chess.WHITE:
                move = mejor_mov(board, model_blancas)
            else:
                move = mejor_mov(board, model_negras)

        board.push(move)
        node = node.add_variation(move)
        ply_count += 1

  
    if not board.is_game_over():
        resultado = "1/2-1/2"
    else:
        resultado = board.result()

    game.headers["Result"] = resultado
    return game, resultado, nombre_blancas, nombre_negras

def main():
   
    custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
    print("Cargando el modelo ...")
    model_v1 = tf.keras.models.load_model("ModeloAjedrez_v8.h5", custom_objects=custom_objects)
    model_v2 = tf.keras.models.load_model("ModeloAjedrez_v7.3.h5")

    num_partidas = 50
    juegos = []


    v1_victorias = 0
    v2_victorias = 0
    tablas = 0

    for i in range(1, num_partidas + 1):
       
        if i % 2 == 0:
            game, resultado, blancas, negras = jugar_partida(model_v1, model_v2, "modelo_v1", "modelo_v2")  # v1 blancas, v2 negras
        else:
            game, resultado, blancas, negras = jugar_partida(model_v2, model_v1, "modelo_v2", "modelo_v1")  # v2 blancas, v1 negras

        print(f"Partida {i}: {resultado} ({blancas} vs {negras})")
        juegos.append(game)

        if resultado == "1-0":
            if blancas == "modelo_v1":
                v1_victorias += 1
            else:
                v2_victorias += 1
        elif resultado == "0-1":
            if negras == "modelo_v1":
                v1_victorias += 1
            else:
                v2_victorias += 1
        else:
            tablas += 1

    # Guardar todas las partidas en un archivo PGN
    with open("partidas_2.pgn", "w") as f:
        for game in juegos:
            exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
            pgn_str = game.accept(exporter)
            f.write(pgn_str + "\n\n")

    print("\nResultados finales:")
    print(f"Modelo V1 victorias: {v1_victorias}")
    print(f"Modelo V2 victorias: {v2_victorias}")
    print(f"Tablas: {tablas}")

if __name__ == "__main__":
    main()

