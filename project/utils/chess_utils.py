import chess
def isDraw(board : chess.Board) -> bool:
    return board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_seventyfive_moves()
    