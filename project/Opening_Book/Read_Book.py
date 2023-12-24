import chess
import chess.polyglot
import random
import os

class Read_Book:
    def __init__(self):
        self.Path = os.path.join(os.path.dirname(__file__), '../Opening_Book/baron30.bin')
        #  self.Path = "C:\Data\5-AI\GitHub\5AI_LAB1_SEARCH\project\Opening_Book\baron30.bin"

    def Opening(self, board):
        moves = []
        opening_move = None
        try:
            with chess.polyglot.open_reader(self.Path) as reader:
                for entry in reader.find_all(board):
                    moves.append(entry.move)
        except Exception as e:
            print(f"Error opening file: {e}")
        if moves:
            opening_move = random.choice(moves)
        else:
            opening_move = None

        return opening_move
