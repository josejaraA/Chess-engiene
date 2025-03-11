import chess
import chess.engine
import chess.pgn
import random


TIME_L = 0.1
NUM_PARTIDAS = 500


pgn_filename = "partidas_stockfish.pgn"
STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)



with open(pgn_filename, "w", encoding="utf-8") as pgn_file:
   
    for partida_num in range(1, NUM_PARTIDAS + 1):
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = "Self-Play con movimientos aleatorios iniciales"
        game.headers["Site"] = "Local"
        game.headers["Round"] = str(partida_num)
        game.headers["White"] = "Stockfish17"
        game.headers["Black"] = "Stockfish17"

        node = game

        for _ in range(6):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            random_move = random.choice(legal_moves)
            board.push(random_move)
            node = node.add_variation(random_move)
            
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            random_move = random.choice(legal_moves)
            board.push(random_move)
            node = node.add_variation(random_move)

        while not board.is_game_over(claim_draw=True):
            result = engine.play(board, chess.engine.Limit(time=TIME_L))
            move = result.move
            board.push(move)
            node = node.add_variation(move)

        game.headers["Result"] = board.result(claim_draw=True)

        pgn_file.write(str(game))
        pgn_file.write("\n\n")
        print(f"Partida {partida_num} completada: {game.headers['Result']}")
        
engine.quit()

print(f"Se han guardado {NUM_PARTIDAS} partidas en {pgn_filename}.")