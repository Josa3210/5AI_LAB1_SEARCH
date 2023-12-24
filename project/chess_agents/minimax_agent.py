
import chess
from project.chess_agents.agent import Agent
import time
from project.chess_utilities.utility import Utility
from project.Opening_Book.Read_Book import Read_Book

class MiniMaxAgent(Agent):

    def __init__(self, utility: Utility, time_limit_move: float) -> None:
        super().__init__(utility, time_limit_move)
        self.name = "ML Bot"
        self.author = "Neural Network Ninjas"
        self.beginning = True
        self.Read_Book = Read_Book()

    def calculate_move(self, board: chess.Board) -> chess.Move:
        # TODO Minimax:
        #   - Move ordering
        #   - iterative deepening
        #   - possible extras:
        #   - Alpha beta (requirement)
        #   use self.utility.board_value(board) to get an evaluation of the board.
        Start_time = time.time()
        Current_depth = 3
        best_move = None
        opening_move = None

        #  If the agent is playing as black, the utility values are flipped (negative-positive)
        flip_value = 1 if board.turn == chess.WHITE else -1

        #  If begin game we use opening playbook
        opening_move = self.Read_Book.Opening(board)
        if opening_move is None:
            pass
        else:
            return opening_move

        best_utility = float("-inf") if board.turn == chess.WHITE else float("inf")
        #       for Current_depth in range(1, depth + 1): #Iterative Deepening

        move, utility = self.MiniMax(board, Current_depth, float("-inf"), float("inf"), flip_value, Start_time)
        #  if (flip_value == 1 and utility > best_utility) or (flip_value == -1 and utility < best_utility):
        best_move = move  # best move update als utility beter is dan vorige best_utility
        #    best_utility = utility
        return best_move

    def MiniMax(self, board, depth, alpha, beta, flip_value, Start_time) -> tuple[chess.Move, float]:
        if depth == 0 or board.is_game_over():
            return board.peek(), flip_value * self.utility.board_value(board)

        if flip_value == 1:  # When white -> Maximizing
            return self.Max_value(board, depth, alpha, beta, flip_value, Start_time)
        else:  # When Black -> Minimizing
            return self.Min_Value(board, depth, alpha, beta, flip_value, Start_time)

    def Max_value(self, board, depth, alpha, beta, flip_value, Start_time) -> tuple[chess.Move, float]:
        best_move = None
        best_utility = float("-inf")
        # Loop through all legal moves
        for move in board.legal_moves:
            # Check if the maximum calculation time for this move has been reached
            if time.time() - Start_time > self.time_limit_move:
                break
            # Play the move
            board.push(move)
            _, value = self.MiniMax(board, depth - 1, alpha, beta, -flip_value, Start_time)
            board.pop()
            if value > best_utility:
                best_move, best_utility = move, value
            alpha = max(alpha, best_utility)
            if beta <= alpha:
                break

        return best_move, best_utility

    def Min_Value(self, board, depth, alpha, beta, flip_value, Start_time):
        best_move = None
        best_utility = float("inf")
        # Loop through all legal moves
        for move in board.legal_moves:
            # Check if the maximum calculation time fot this move has been reached
            if time.time() - Start_time > self.time_limit_move:
                break
            # Play the move
            board.push(move)
            _, value = self.MiniMax(board, depth - 1, alpha, beta, -flip_value, Start_time)
            board.pop()
            if value < best_utility:
                best_move, best_utility = move, value
            beta = min(beta, best_utility)
            if beta <= alpha:
                break

        return best_move, best_utility