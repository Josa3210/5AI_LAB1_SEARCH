import chess
from project.chess_utilities.utility import Utility
from project.machine_learning.neural_network_heuristic import NeuralNetworkHeuristic


class MachineLearningUtility(Utility):
    def __init__(self, network: NeuralNetworkHeuristic) -> None:
        super().__init__()
        self.network = network
        self.network.eval()

    def board_value(self, board: chess.Board) -> float:
        return self.network.execute(board=board)
