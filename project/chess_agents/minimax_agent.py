
import chess
from project.chess_agents.agent import Agent
import time
from project.chess_utilities.utility import Utility
from project.Opening_Book.Read_Book import Read_Book
from project.utils.chess_utils import hash

class TranspositionEntry:
    def __init__(self, flag: str, value : float, depth : int, sortedMoves : list[chess.Move]) -> None:
        self.flag : str = flag
        self.value = value
        self.depth = depth
        self.sortedMoves : list[chess.Move] = sortedMoves

class MiniMaxAgent(Agent):

    def __init__(self, utility: Utility, time_limit_move: float) -> None:
        super().__init__(utility, time_limit_move)
        self.name = "ML Bot"
        self.author = "Neural Network Ninjas"
        self.beginning = True
        self.Read_Book = Read_Book()
        self.transpositionTable : dict[int, TranspositionEntry] = dict()

    def calculate_move(self, board: chess.Board) -> chess.Move:
        # TODO Minimax:
        #   - Move ordering
        #   - iterative deepening
        #   - possible extras:
        #   - Alpha beta (requirement)
        #   use self.utility.board_value(board) to get an evaluation of the board.
        start_time = time.time()
        #  If begin game we use opening playbook
        opening_move = self.Read_Book.Opening(board)
        if opening_move is not None:
            return opening_move
        
        # best_utility = float("-inf") if board.turn == chess.WHITE else float("inf")
        # #       for Current_depth in range(1, depth + 1): #Iterative Deepening

        # move, utility = self.MiniMax(board, Current_depth, float("-inf"), float("inf"), flip_value, Start_time)
        # #  if (flip_value == 1 and utility > best_utility) or (flip_value == -1 and utility < best_utility):
        # best_move = move  # best move update als utility beter is dan vorige best_utility
        # #    best_utility = utility
        # return best_move
        
        #if white is playing, it is the maximizing player, so we will search for all the moves white can make and calculate the advantage black can get
        flip_value = 1 if board.turn == chess.WHITE else -1
        self.transpositionTable = dict()
        
        possible_moves = board.legal_moves
        print(f"{possible_moves.count()} legal moves were found")
        best_move = None
        depth = 1
        print(time.time() - start_time)
        while (time.time() - start_time < self.time_limit_move):
            value = float("-inf")
            alpha = float("-inf")
            beta = float("inf")
            for index, move in enumerate(self.sortMoves(board)):
                if time.time() - start_time > self.time_limit_move:
                    break
                board.push(move)
                opponent_disadvantage = - self.negaMax(board, depth - 1, -flip_value, start_time, alpha, beta)
                if abs(opponent_disadvantage) == float("inf"):
                    print("Weird result, omitting")
                    continue
                if opponent_disadvantage > value:
                    value = opponent_disadvantage
                    best_move = move
                board.pop()
                alpha = max(alpha, opponent_disadvantage)
                print(f"finished evaluating move {index + 1} out of {possible_moves.count()} to depth {depth}")
            depth += 1
        print(f"Best move with value {flip_value * value}")
        return best_move
    
    def negaMax(self, board: chess.Board, depth : int, color: int, start_time : float, alpha = float("-inf"), beta = float("inf")) -> float:
        """
        When the depth is 0 or the board state it terminal it will return the value of the board multiplied by the color.
            This results in a the returning value being favorable to the color: i.e positive is always good.
        
        Otherwise loop over all moves and calculate each move's advantage for the opposite player. 
        Of all these moves select the one that results in our biggest advantage
        
        So the return value is the value associated with the move resulting in the biggest advantage for the given color.
        """
        board_hash = hash(board)
        alpha_init = alpha
        entry = self.transpositionTable.get(board_hash)
        if entry is not None and entry.depth >= depth:
            print("cache hit")
            if entry.flag == "EXACT":
                return entry.value
            elif entry.flag == "LOWER":
                alpha = max(alpha, entry.value)
            elif entry.flag == "UPPER":
                beta = min(beta, entry.value)
            if alpha >= beta :
                return entry.value
        if depth == 0 or board.is_game_over():
            value = self.utility.board_value(board)
            return color * value
        sortedMoves = entry.sortedMoves if entry is not None else self.sortMoves(board)
        value = float("-inf")
        for move in sortedMoves:
            if time.time() - start_time > self.time_limit_move:
                print("time limited")
                break
            board.push(move)
            value = max(value, -self.negaMax(board, depth - 1, -color, start_time, -beta, - alpha)) # selects the disadvantage for the other player if bigger then largest disadvantage
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
            
        if entry is None:
            entry = TranspositionEntry("", None, None, None)
        entry.value = value
        if value <= alpha_init:
            entry.flag = "UPPER"
        elif value >= beta:
            entry.flag = "LOWER"
        else:
            entry.flag = "EXACT"
        entry.depth = depth
        entry.sortedMoves = sortedMoves
        self.transpositionTable[board_hash] = entry
        return value

            
    def sortMoves(self, board: chess.Board) -> list[chess.Move]:
        movesWithEval: list[tuple[chess.Move, float]] = []
        for move in board.legal_moves:
            board.push(move)
            movesWithEval.append((move, self.utility.board_value(board)))
            board.pop()
        movesWithEval.sort(key= lambda x: x[1], reverse=board.turn == chess.WHITE)
        return [moveWithEval[0] for moveWithEval in movesWithEval]

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