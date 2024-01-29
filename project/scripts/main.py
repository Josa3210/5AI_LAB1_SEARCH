import torch
from torch import nn
import torch.cuda

from project.chess_agents.minimax_agent import MiniMaxAgent
from project.chess_engines.uci_engine import UciEngine
from project.chess_utilities.ml_utility import MachineLearningUtility, StockfishUtility

from project.chess_utilities.utility import Utility
from project.machine_learning.neural_network_heuristic import NeuralNetworkHeuristic, CanCaptureHeuristic, WorldViewHeuristic

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    network: nn.Module = WorldViewHeuristic()
    network: NeuralNetworkHeuristic =torch.load("project\\data\\models\\Best") 
    network = network.to(device)
    utility: Utility = MachineLearningUtility(network)
    # utility: Utility = StockfishUtility()
    agent: MiniMaxAgent = MiniMaxAgent(utility=utility, time_limit_move=5.0)
    engine: UciEngine = UciEngine(name="ML bot", author="Neural Network Ninjas", agent=agent)
    engine.engine_operation()
