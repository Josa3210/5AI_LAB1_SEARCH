
import chess
from project.chess_agents.agent import Agent


class MiniMaxAgent(Agent):
    
    def calculate_move(self, board: chess.Board) -> chess.Move:
        # TODO Minimax:
        # - possible extras:
        #   - Alpha beta (requirement)
        #   - Move ordering
        #   - iterative deepening
        # use self.utility.board_value(board) to get an evaluation of the board.
        return list(board.legal_moves)[0]