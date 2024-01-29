import chess
import chess.engine
from project.chess_utilities.utility import Utility
from project.machine_learning.neural_network_heuristic import NeuralNetworkHeuristic


class MachineLearningUtility(Utility):
    def __init__(self, network: NeuralNetworkHeuristic) -> None:
        super().__init__()
        self.network = network
        self.network.eval()

    def board_value(self, board: chess.Board) -> float:
        return self.network.execute(board=board)

class StockfishUtility(Utility):
    def __init__(self) -> None:
        super().__init__()
        self.engine = chess.engine.SimpleEngine.popen_uci("project/chess_engines/stockfish/stockfish-windows-x86-64-avx2.exe")
        
    def board_value(self, board: chess.Board) -> float:
        score = self.engine.analyse(board, limit=chess.engine.Limit(depth=1))["score"].white().score()
        return score/100.0 if score is not None else 0.0