from abc import ABC
import chess
from math import tanh
import chess.engine

"""A generic utility class"""
class Utility(ABC):

    def __init__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(
                    "project/chess_engines/stockfish/stockfish-windows-x86-64-avx2.exe")
    # Determine the value of the current board position (high is good for white, low is good for black, 0 is neutral)
    def board_value(self, board: chess.Board) -> float:
        info = self.engine.analyse(board, limit=chess.engine.Limit(time=0.1, depth=3))
        pawnScore = info["score"].white().score(mate_score=100000) / 100.0
        return pawnScore
