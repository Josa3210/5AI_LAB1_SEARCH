import torch
from torch import nn

from project.chess_agents.minimax_agent import MiniMaxAgent
from project.chess_engines.uci_engine import UciEngine
from project.chess_utilities.ml_utility import MachineLearningUtility

from project.chess_utilities.utility import Utility
from project.machine_learning.neural_network_heuristic import NeuralNetworkHeuristic, CanCaptureHeuristic

if __name__ == "__main__":
    model: nn.Module = CanCaptureHeuristic(256, 128, 0, nn.ReLU(), 0.5)
    network: NeuralNetworkHeuristic = model.load_state_dict(torch.load("project\\data\\models\\Best"))  # TODO load the neural network from the disk
    utility: Utility = MachineLearningUtility(network)
    # utility: Utility = Utility()
    agent: MiniMaxAgent = MiniMaxAgent(utility=utility, time_limit_move=5.0)
    engine: UciEngine = UciEngine(name="ML bot", author="Neural Network Ninjas", agent=agent)
    engine.engine_operation()
