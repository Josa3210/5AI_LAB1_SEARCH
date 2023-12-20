import string
from abc import abstractmethod

import chess
from torch import nn
import torch
import numpy


class NeuralNetworkHeuristic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def execute(self, board: chess.Board) -> float:
        pass

    def featureExtraction(board: chess.Board) -> torch.Tensor:
        pass


class SimpleNeuralNetworkHeuristic(NeuralNetworkHeuristic):
    def __init__(self) -> None:
        super().__init__()
        n_inputs: int = 65
        nL1: int = 32
        nL2: int = 8
        n_output: int = 1
        self.layers: nn.Sequential = nn.Sequential(nn.Linear(n_inputs, nL1), nn.ReLU(), nn.Linear(nL1, nL2), nn.ReLU(), nn.Linear(nL2, n_output), nn.Tanh())  # Todo define a the necessary layers https://youtu.be/ORMx45xqWkA?t=111

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

    def getName(self) -> string:
        return "SimpleHeuristic"

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


class CanCaptureHeuristic(NeuralNetworkHeuristic):
    def __init__(self, nL1: int, nL2: int, nL3: int, activation: nn.modules.activation, dropoutRate: float) -> None:
        super().__init__()
        super().__init__()

        self.nInput = 64 * 12 * 9
        self.nOutput = 1

        self.nL1 = nL1
        self.nL2 = nL2
        self.nL3 = nL3

        self.activation = activation
        self.dropoutRate = dropoutRate

        if self.nL1 != 0:
            self.layers: nn.Sequential = nn.Sequential(
                nn.Linear(self.nInput, self.nL1),
                self.activation,
                nn.Dropout(self.dropoutRate),
                nn.Linear(self.nL1, self.nOutput),
                nn.Tanh()
            )
        elif self.nL2 != 0:
            self.layers: nn.Sequential = nn.Sequential(
                nn.Linear(self.nInput, self.nL1),
                self.activation,
                nn.Dropout(self.dropoutRate),
                nn.Linear(self.nL1, self.nL2),
                self.activation,
                nn.Dropout(self.dropoutRate),
                nn.Linear(self.nL2, self.nOutput),
                nn.Tanh()
            )
        elif self.nL3 != 0:
            self.layers: nn.Sequential = nn.Sequential(
                nn.Linear(self.nInput, self.nL1),
                self.activation,
                nn.Dropout(self.dropoutRate),
                nn.Linear(self.nL1, self.nL2),
                self.activation,
                nn.Dropout(self.dropoutRate),
                nn.Linear(self.nL2, self.nL3),
                self.activation,
                nn.Dropout(self.dropoutRate),
                nn.Linear(self.nL3, self.nOutput),
                nn.Tanh()
            )

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
        The input will be a set of (curr_pos, piece_type, am_can_capture)
        Input size will be: 64 * 13 * 9 = 5760
        - 64 board positions    -> starting from A1 (left white rook) to H7 (right black rook)
        - 13 piece types        -> 0: no piece, 1: white pawn, 2: white bishop, 3: white knight, 4: white rook, 5: white queen, 6: white king, 7-12: same but black pieces
        - 9 captures states     -> the amount of pieces that can be captured by the piece (0 is none can be taken, +1 for every direction a piece can be captured)
        """

        # Initialize the empty tensor and empty dictionary
        attackersDict = dict()  # key = position, value = amount of pieces it can capture
        features: torch.Tensor = torch.zeros([64, 12, 9], dtype=torch.float32)

        # Go over all the positions
        for square in chess.SQUARES:
            attackedPiece = board.piece_at(square)
            # Check if there is a piece on the current square
            if attackedPiece is not None:
                # Determine the color of the attacking piece
                attackingColor = not attackedPiece.color

                # Look at all the pieces of the attackingColor that can attack the current position
                attackers = board.attackers(attackingColor, square)

                # If there is an attacker, set that position in the dict +1 because it can attack a piece of the other color
                for attackerSquare in attackers:
                    if attackerSquare not in attackersDict.keys():
                        attackersDict[attackerSquare] = 1
                    else:
                        attackersDict[attackerSquare] += 1

        for square in chess.SQUARES:
            currPiece = board.piece_at(square)
            if currPiece is not None:
                # Get piece pieceColor
                if currPiece.color == board.turn:
                    pieceColor = 0
                else:
                    pieceColor = 5

                # Get the piece value: this will be the value of the piece + 5 if it is a black piece
                # We do -1 because the Tensor starts at 0 (white pawn = 0)
                pieceVal = currPiece.piece_type + pieceColor - 1

                if square not in attackersDict.keys():
                    pieceAttacks = 0
                else:
                    pieceAttacks = attackersDict[square]

                features[square][pieceVal][pieceAttacks] = 1

        return torch.flatten(features)

    def getName(self) -> string:
        return "Optimizable_CanCaptureHeuristic"
