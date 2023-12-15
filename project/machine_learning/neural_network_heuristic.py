import string

import chess
from torch import nn
import torch
import numpy


class SimpleNeuralNetworkHeuristic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        n_inputs: int = 65
        n_L1: int = 32
        n_L2: int = 8
        n_output: int = 1
        self.layers: nn.Sequential = nn.Sequential(nn.Linear(n_inputs, n_L1), nn.ReLU(), nn.Linear(n_L1, n_L2), nn.ReLU(), nn.Linear(n_L2, n_output), nn.Tanh())  # Todo define a the necessary layers https://youtu.be/ORMx45xqWkA?t=111

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This is the function we call to train the model. This should not be called
        By the end user. they should call self.execute()
        """
        output: torch.Tensor = self.layers.forward(x)
        return output

    def execute(self, board: chess.Board) -> float:
        """
        This is the end user's function to use for the neural network.
        """
        # Extract features from board
        features: torch.Tensor = self.featureExtraction(board)
        output: float = self.forward(features)[0]
        return output

    def featureExtraction(board: chess.Board) -> torch.Tensor:
        """
        This takes a board and converts it to the input of our neural network.
        """
        # extract position from board and structure them in array of 64 spaces
        # White > 0, Black < 0, 0 = no piece
        # Pion = 1, Loper = 2, Knight = 3, Toren = 4, Queen = 5, King = 6
        fenString: string = board.fen()

        features: torch.Tensor = torch.zeros([65], dtype=torch.float32)

        color: int = 1
        piece: int = 0
        pos: int = 0

        # filter fenString
        positions: string = fenString.split()[0]
        positions = positions.replace('/', "")

        turnString: string = fenString.split()[1]
        turn: int

        # Get turn out of fenString
        if turnString == "w":
            turn = 10
        else:
            turn = -10

        # Get the position out of fenString
        for char in positions:
            # Number means x empty spaces
            if char.isnumeric():
                for i in range(int(char)):
                    features[pos] = 0
                    pos += 1

            # Not a number means there is a piece
            else:

                # Get color of piece
                if char.islower():
                    color = -1
                else:
                    color = 1

                # Get piece value
                match char.lower():
                    case 'p':
                        piece = 1
                    case 'b':
                        piece = 2
                    case 'n':
                        piece = 3
                    case 'r':
                        piece = 4
                    case 'q':
                        piece = 5
                    case 'k':
                        piece = 6
                    case _:
                        piece = 0

                features[pos] = color * piece
                pos += 1

        features[pos] = turn

        return features


class CanCaptureNeuralNetworkHeuristic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        n_inputs: int = 5760
        n_L1: int = 256
        n_L2: int = 8
        n_output: int = 1
        self.layers: nn.Sequential = nn.Sequential(
            nn.Linear(n_inputs, n_L1),
            nn.ReLU(),
            nn.Linear(n_L1, n_L2),
            nn.ReLU(),
            nn.Linear(n_L2, n_output),
            nn.Tanh()
        )  # Todo define a the necessary layers https://youtu.be/ORMx45xqWkA?t=111

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This is the function we call to train the model. This should not be called
        By the end user. they should call self.execute()
        """
        output: torch.Tensor = self.layers.forward(x)
        return output

    def execute(self, board: chess.Board) -> float:
        """
        This is the end user's function to use for the neural network.
        """
        # Extract features from board
        features: torch.Tensor = self.featureExtraction(board)
        inputT: torch.Tensor = torch.flatten(features)
        output: float = self.forward(inputT)[0]
        return output

    def featureExtraction(board: chess.Board) -> torch.Tensor:
        """
        This takes a board and converts it to the input of our neural network.
        The input will be a set of (curr_pos, piece_type, am_can_capture)
        Input size will be: 64 * 10 * 9 = 5760
        """

        fenString: string = board.fen()

        features: torch.Tensor = torch.zeros([64, 10, 9], dtype=torch.float32)
        attackersDict = dict()

        # filter fenString
        positions: string = fenString.split()[0]
        positions = positions.replace('/', "")

        for pos in range(63):
            attackers = board.attackers(chess.WHITE, pos)
            for attacker in attackers:
                if pos not in attackersDict.keys():
                    attackersDict[pos] = 1
                else:
                    attackersDict[pos] += 1

        pos = 0
        for char in positions:
            if not char.isnumeric():
                # Get piece pieceColor
                if char.islower():
                    pieceColor = -1
                else:
                    pieceColor = 1

                # Get piece value
                match char.lower():
                    case 'p':
                        pieceVal = 1
                    case 'b':
                        pieceVal = 2
                    case 'n':
                        pieceVal = 3
                    case 'r':
                        pieceVal = 4
                    case 'q':
                        pieceVal = 5
                    case 'k':
                        pieceVal = 6
                    case _:
                        pieceVal = 0
                piece = pieceVal * pieceColor
                if pos not in attackersDict.keys():
                    pieceAttacks = 0
                else:
                    pieceAttacks = attackersDict[pos]

                features[pos][piece][pieceAttacks] = 1
            pos += 1
        return features
