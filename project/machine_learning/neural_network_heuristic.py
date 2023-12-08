import chess
from torch import nn

class NeuralNetworkHeuristic(nn.Module):
    def __init__(self) -> None:
        self.layers : nn.Sequential = None #Todo define a the necessary layers https://youtu.be/ORMx45xqWkA?t=111
    
    def forward(self, x: chess.Board) -> float:
        pass