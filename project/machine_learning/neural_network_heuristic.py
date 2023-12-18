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


class CanCaptureNeuralNetworkHeuristic(NeuralNetworkHeuristic):
    def __init__(self) -> None:
        """
        Initializes the amount if input and output nodes
        - Input is derived from the heuristic (look at heuristic for more explanation):
            - 64 positions
            - 13 pieces
            - 9 possible capture states
        - Output is 1 node (regression)
            - 1     -> advantage for white
            - 0     -> draw
            - -1    -> advantage for black
        """
        super().__init__()

        self.nInput = 64 * 13 * 9
        self.nOutput = 1

        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This is the function we call to train the model. This should not be called
        By the end user. they should call self.execute()
        """
        output: torch.Tensor = self.layers.forward(x)
        return output

    @abstractmethod
    def getName(self) -> string:
        """
        This function will give the model a name with some characteristics.
        This name will be used when saving the model so the different models can be easily found
        """
        pass

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
        Input size will be: 64 * 13 * 9 = 5760
        - 64 board positions    -> starting from A1 (left white rook) to H7 (right black rook)
        - 13 piece types        -> 0: no piece, 1: white pawn, 2: white bishop, 3: white knight, 4: white rook, 5: white queen, 6: white king, 7-12: same but black pieces
        - 9 captures states     -> the amount of pieces that can be captured by the piece (0 is none can be taken, +1 for every direction a piece can be captured)
        """

        # Initialize the empty tensor and empty dictionary
        attackersDict = dict()  # key = position, value = amount of pieces it can capture
        features: torch.Tensor = torch.zeros([64, 13, 9], dtype=torch.float32)

        # filter fenString
        fenString: string = board.fen()
        positions: string = fenString.split()[0]
        positions = positions.replace('/', "")

        # Go over all the positions
        for pos in range(63):

            # Check if there is a piece on the current square
            attackedPiece: chess.Piece = board.piece_at(pos)
            if attackedPiece is not None:

                # Determine the color of the attacking piece
                if attackedPiece.color == chess.WHITE:
                    attackingColor = chess.BLACK
                else:
                    attackingColor = chess.WHITE

                # Look at all the pieces of the attackingColor that can attack the current position
                attackers = board.attackers(attackingColor, pos)

                # If there is an attacker, set that position in the dict +1 because it can attack a piece of the other color
                for attacker in attackers:
                    if attacker not in attackersDict.keys():
                        attackersDict[attacker] = 1
                    else:
                        attackersDict[attacker] += 1

        pos = 0
        for char in positions:
            if not char.isnumeric():
                # Get piece pieceColor
                if char.islower():
                    pieceColor = 5
                else:
                    pieceColor = 0

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
                piece = pieceVal + pieceColor
                if pos not in attackersDict.keys():
                    pieceAttacks = 0
                else:
                    pieceAttacks = attackersDict[pos]

                features[pos][piece][pieceAttacks] = 1
            pos += 1
        return torch.flatten(features)


class OptimizeLayers(CanCaptureNeuralNetworkHeuristic):
    def __init__(self, nL1: int, nL2: int, nL3: int) -> None:
        super().__init__()

        self.nL1 = nL1
        self.nL2 = nL2
        self.nL2 = nL3

        if nL3 != 0:
            self.layers: nn.Sequential = nn.Sequential(
                nn.Linear(self.nInput, self.nL1),
                nn.ReLU(),
                nn.Linear(self.nL1, self.nL2),
                nn.ReLU(),
                nn.Linear(self.nL2, self.nL3),
                nn.ReLU(),
                nn.Linear(self.nL3, self.nOutput),
                nn.Tanh()
            )  # Todo define a the necessary layers https://youtu.be/ORMx45xqWkA?t=111
        else:
            self.layers: nn.Sequential = nn.Sequential(
                nn.Linear(self.nInput, nL1),
                nn.ReLU(),
                nn.Linear(nL1, nL2),
                nn.ReLU(),
                nn.Linear(nL2, self.nOutput),
                nn.Tanh()
            )  # Todo define a the necessary layers https://youtu.be/ORMx45xqWkA?t=111

    def getName(self) -> string:
        return f"OptimizeLayers_canCaptureHeuristic_{self.nL1}_{self.nL2}_{self.nL3}"


class OptimizeActivationReLu(CanCaptureNeuralNetworkHeuristic):
    def __init__(self, nL1: int, nL2: int, nL3: int) -> None:
        super().__init__()

        self.nL1 = nL1
        self.nL2 = nL2
        self.nL2 = nL3

        if nL3 != 0:
            self.layers: nn.Sequential = nn.Sequential(
                nn.Linear(self.nInput, self.nL1),
                nn.ReLU(),
                nn.Linear(self.nL1, self.nL2),
                nn.ReLU(),
                nn.Linear(self.nL2, self.nL3),
                nn.ReLU(),
                nn.Linear(self.nL3, self.nOutput),
                nn.Tanh()
            )  # Todo define a the necessary layers https://youtu.be/ORMx45xqWkA?t=111
        else:
            self.layers: nn.Sequential = nn.Sequential(
                nn.Linear(self.nInput, nL1),
                nn.ReLU(),
                nn.Linear(nL1, nL2),
                nn.ReLU(),
                nn.Linear(nL2, self.nOutput),
                nn.Tanh()
            )  # Todo define a the necessary layers https://youtu.be/ORMx45xqWkA?t=111

    def getName(self) -> string:
        return f"OptimizeActivation_canCaptureHeuristic_ReLu"


class OptimizeActivationLeakyReLu(CanCaptureNeuralNetworkHeuristic):
    def __init__(self, nL1: int, nL2: int, nL3: int) -> None:
        super().__init__()

        self.nL1 = nL1
        self.nL2 = nL2
        self.nL2 = nL3

        if nL3 != 0:
            self.layers: nn.Sequential = nn.Sequential(
                nn.Linear(self.nInput, self.nL1),
                nn.LeakyReLU(),
                nn.Linear(self.nL1, self.nL2),
                nn.LeakyReLU(),
                nn.Linear(self.nL2, self.nL3),
                nn.LeakyReLU(),
                nn.Linear(self.nL3, self.nOutput),
                nn.Tanh()
            )  # Todo define a the necessary layers https://youtu.be/ORMx45xqWkA?t=111
        else:
            self.layers: nn.Sequential = nn.Sequential(
                nn.Linear(self.nInput, nL1),
                nn.LeakyReLU(),
                nn.Linear(nL1, nL2),
                nn.LeakyReLU(),
                nn.Linear(nL2, self.nOutput),
                nn.Tanh()
            )  # Todo define a the necessary layers https://youtu.be/ORMx45xqWkA?t=111

    def getName(self) -> string:
        return f"OptimizeActivation_canCaptureHeuristic_LeakyReLu"
