
from project.chess_agents.minimax_agent import MiniMaxAgent
from project.chess_engines.uci_engine import UciEngine
#  from project.chess_utilities.ml_utility import MachineLearningUtility
#  from project.machine_learning.neural_network_heuristic import NeuralNetworkHeuristic
from project.chess_utilities.utility import Utility


if __name__ == "__main__":
  #  network : NeuralNetworkHeuristic = None # TODO load the neural network from the disk
  #  utility : MachineLearningUtility = MachineLearningUtility(network)
    utility : Utility = Utility()
    agent : MiniMaxAgent = MiniMaxAgent(utility=utility, time_limit_move=5.0)
    engine : UciEngine = UciEngine(name="ML bot", author="Neural Network Ninjas", agent=agent)
    engine.engine_operation()
