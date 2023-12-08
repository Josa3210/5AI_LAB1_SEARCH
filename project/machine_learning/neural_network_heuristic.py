import chess
from torch import nn
import torch

class NeuralNetworkHeuristic(nn.Module):
    def __init__(self) -> None:
        self.layers : nn.Sequential = None #Todo define a the necessary layers https://youtu.be/ORMx45xqWkA?t=111
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This is the function we call to train the model. This should not be called
        By the end user. they should call self.execute()
        """
        
        # TODO
        pass
    
    def execute(self, board: chess.Board) -> float:
        """
        This is the end user's function to use for the neural network.
        """
        # TODO
        pass
    
    def featureExtraction(board: chess.Board) -> torch.Tensor:
        """
        This takes a board and converts it to the input of our neural network.
        """
        # TODO