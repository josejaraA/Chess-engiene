import chess.pgn
import chess.engine
import pandas as pd
from tqdm import tqdm

# Configura la ruta del motor Stockfish
STOCKFISH_PATH = "D:/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"


def analyze_pgn(file_path, output_csv, depth=15):
    """Analiza un archivo PGN y extrae la distribución de aperturas, errores y blunders por color, y la evolución de la evaluación."""
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    games_data = []
    
    with open(file_path) as pgn:
        game_count = 0
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            
            game_count += 1
            board = game.board()
            opening = game.headers.get("ECO", "Desconocido")
            white_elo = game.headers.get("WhiteElo", "?")
            black_elo = game.headers.get("BlackElo", "?")
            result = game.headers.get("Result", "?")
            
            errors_white = 0
            errors_black = 0
            blunders_white = 0
            blunders_black = 0
            eval_history = []
            num_moves = 0
            last_eval = None
            
            for move in game.mainline_moves():
                board.push(move)
                num_moves += 1
                eval_info = engine.analyse(board, chess.engine.Limit(depth=depth))
                
                if "score" in eval_info:
                    eval_score = eval_info["score"].relative.score(mate_score=10000)  # Convierte mate a valor alto
                    eval_history.append(eval_score)
                    
                    if last_eval is not None:
                        delta = last_eval - eval_score if board.turn == chess.BLACK else eval_score - last_eval
                        
                        if board.turn == chess.BLACK:  # El último movimiento fue de blancas
                            if delta >= 300:
                                blunders_white += 1
                            elif delta >= 100:
                                errors_white += 1
                        else:  # El último movimiento fue de negras
                            if delta >= 300:
                                blunders_black += 1
                            elif delta >= 100:
                                errors_black += 1
                    
                    last_eval = eval_score
            
            games_data.append([opening, white_elo, black_elo, result, num_moves, errors_white, errors_black, blunders_white, blunders_black, eval_history])
            
            if game_count % 100 == 0:
                print(f"{game_count} partidas procesadas...")
    
    engine.quit()
    
    df = pd.DataFrame(games_data, columns=["Apertura", "ELO_Blancas", "ELO_Negras", "Resultado", "Num_Movimientos", "Errores_Blancas", "Errores_Negras", "Blunders_Blancas", "Blunders_Negras", "Evaluacion_Historial"])
    df.to_csv(output_csv, index=False)
    print(f"Datos guardados en {output_csv}")

# Ejecutar análisis sobre un archivo PGN
pgn_file = "data_entrenamiento/datasets.pgn"  # Reemplázalo con el nombre de tu archivo
output_file = "analisis_partidas_1.csv"
analyze_pgn(pgn_file, output_file, depth=15)

