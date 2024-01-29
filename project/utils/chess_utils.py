import random
import chess
from chess.polyglot import ZobristHasher
def isDraw(board : chess.Board) -> bool:
    return board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_seventyfive_moves()
   
hasher = ZobristHasher(array=random.sample(range(pow(2, 30)), 1024))
def hash(board: chess.Board) -> int:
    global hasher
    return hasher.hash_board(board)
    