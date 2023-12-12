import chess
import chess.pgn
import chess.engine
import os

from project.machine_learning.parsing import DataParser

def test_space():
    pgn = open("project/data/Carlsen.pgn")
    game : chess.pgn.Game = chess.pgn.read_game(pgn)
    board : chess.Board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        print(board)
    print(game)
    
def test_read_multiple():
    pgn = open("project/data/kasparov-deep-blue-1997.pgn")
    game = chess.pgn.read_game(pgn)
    counter = 1
    while game is not None:
        game = chess.pgn.read_game(pgn)
        print(game)
        counter += 1
    print(counter)
    
def test_evaluate_position():
    board = chess.Board("rnbqkbnr/ppp2ppp/4p3/3p4/4P3/3P4/PPP2PPP/RNBQKBNR b KQkq d6 0 3")
    with chess.engine.SimpleEngine.popen_uci("project/chess_engines/stockfish/stockfish-windows-x86-64-avx2.exe") as engine:
        info = engine.analyse(board, limit=chess.engine.Limit(time=1.0, depth=4))
        print(info["score"].white().score()/100.0)
        
def test_evaluate_position_using_stockFish():
    board = chess.Board("rnbqkbnr/ppp2ppp/4p3/3p4/4P3/3P4/PPP2PPP/RNBQKBNR b KQkq d6 0 3")
    value = DataParser.evaluateUsingStockFish(board)
    assert value == -0.32
    
def test_parser():
    parser = DataParser("project/data/kasparov-deep-blue-1997.pgn")
    parser.parse()
    assert os.path.exists("project/data/kasparov-deep-blue-1997.pgn.cache")
    for board, value in parser.values():
        assert "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R" in board.fen()
        assert value == 0.22
        print(board, value)
        break