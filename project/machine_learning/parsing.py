from copy import deepcopy
from math import tanh
from typing import Generator

import chess
import chess.pgn
import chess.engine
import chess.polyglot
import os
from torch.utils.data import Dataset, DataLoader
from project.machine_learning.neural_network_heuristic import NeuralNetworkHeuristic
from project.utils.chess_utils import isDraw
import torch

boardTensor = torch.Tensor
evaluationTensor = torch.Tensor
chessData = tuple[chess.Board, float]
chessDataTensor = tuple[boardTensor, torch.Tensor]
chessDataBatch = list[chessDataTensor]


class ChessDataSet(Dataset):
    def __init__(self, dataGenerator: Generator[chessData, None, None]) -> None:
        super().__init__()
        self.data: list[chess.Board] = []
        self.labels: list[float] = []
        for board, value in dataGenerator:
            self.data.append(board)
            self.labels.append(value)

    def __getitem__(self, index) -> chessDataTensor:
        return NeuralNetworkHeuristic.featureExtraction(self.data[index]), torch.tensor(self.labels[index])

    def __len__(self) -> int:
        return len(self.labels)


class ChessDataLoader():
    def __init__(self, data_parsers: list["DataParser"], batch_size: int = 32, heuristic = NeuralNetworkHeuristic) -> None:
        self.batch_size: int = batch_size
        self.data_size: int = sum([parser.size for parser in data_parsers])
        self.data: list[chessDataBatch] = None
        self.data_parsers = data_parsers
        self.heuristic = heuristic
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def __iter__(self) -> Generator[chessDataTensor, None, None]:
        if self.data != None:
            yield from self.data
            return
        currentBatch: list[tuple[chess.Board, float]] = []
        features_device , evaluations_device = [], []
        for parser in self.data_parsers:
            for chessData in parser.values():
                currentBatch.append(chessData)
                if len(currentBatch) is self.batch_size:
                    features, evaluations = self.chessDataToTensor(currentBatch)
                    del features_device
                    del evaluations_device
                    features_device = features.to(self.device)
                    evaluations_device = evaluations.to(self.device)
                    yield features_device, evaluations_device
                    currentBatch = []
            if len(currentBatch) != 0:
                features, evaluations = self.chessDataToTensor(currentBatch)
                del features_device
                del evaluations_device
                features_device = features.to(self.device)
                evaluations_device = evaluations.to(self.device)
                yield features_device, evaluations_device

    def __len__(self):
        return self.data_size

    def chessDataToTensor(self, chessData: list[chessData]) -> chessDataTensor:
        boardFeatures: list[boardTensor] = []
        evaluations: list[torch.Tensor] = []
        for board, evaluation in chessData:
            boardFeature = self.heuristic.featureExtraction(board)
            boardFeatures.append(boardFeature)
            evaluations.append(torch.tensor(evaluation))
        evaluations = torch.stack(evaluations).unsqueeze(1)
        features =  torch.stack(boardFeatures)
        return features, evaluations

class DataParser():
    """
    This class is the place to crunch large datasets into useable data for training.
    """

    def __init__(self, filePath: str) -> None:
        self.filePath = filePath
        self.cachedFile = filePath + ".cache"
        self.dataSet: ChessDataSet = None
        self.size = 0

    def readGame(self):
        with open(self.filePath, encoding='utf-8') as file:
            game = chess.pgn.read_game(file)
            while game is not None:
                game = chess.pgn.read_game(file)
                if game is not None:
                    yield game

    def parse(self, overwriteCache=False) -> None:
        """
        Call this method to parse the file and hold it in memory.
        if overwriteCache is false it will look if the calculations were already done.
        """
        if not overwriteCache and os.path.exists(self.cachedFile):
            print(f"Found previous data, values are already available")
            with open(self.cachedFile, 'rb') as file:
                self.size = sum(1 for _ in file)
            return

        print(f"Reading file: {self.filePath}")
        boards: list[chess.Board] = []
        i: int = 0
        reader = chess.polyglot.MemoryMappedReader("project/Opening_Book/baron30.bin")
        openingDrops = 0
        mateDrops = 0
        drawDrops = 0
        for game in self.readGame():
            board: chess.Board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if reader.get(board) is not None:
                     openingDrops += 1
                     continue
                elif board.is_checkmate():
                    mateDrops += 1
                    continue
                elif isDraw(board):
                    drawDrops += 1
                    continue
                boards.append(deepcopy(board))
            if (i + 1) % 100 == 0:
                print(f"Read {i + 1} games for extracting positions", end='\r')
            i += 1
        positionCount = len(boards) - openingDrops - mateDrops - drawDrops
        print(f"""
    Read in {positionCount} different positions
    Omitted positions:
    - {openingDrops} were opening moves
    - {mateDrops} were mates
    - {drawDrops} were draws
              """)
        with open(self.cachedFile, 'w') as cacheFile:
            with chess.engine.SimpleEngine.popen_uci("project/chess_engines/stockfish/stockfish-windows-x86-64-avx2.exe") as engine:
                counter = 0
                for board in boards:
                    fenString = board.fen()
                    value = DataParser.evaluateUsingStockFish(board, engine=engine)
                    if value is None:
                        continue
                    cacheFile.write(f"{fenString},{value}\n")
                    counter += 1
                    if counter % 500 == 0:
                        print(f"Read in and evaluated {counter} out of {positionCount} positions.", end='\r')
                        cacheFile.flush()
                self.size = counter
        print(f"Evaluated all {positionCount} positions, data is now accessible via self.values")

    def values(self) -> Generator[tuple[chess.Board, float], None, None]:
        """
        This function generates an iterable: https://www.youtube.com/watch?v=HnggP09mKpM
        Use this in a for loop when training the NN.
        """
        with open(self.cachedFile) as data:
            lines = data.readlines()
            totalLines = len(lines)
            linesRead = 0
            percentage = 0
            for line in lines:
                try:
                    linesRead += 1
                    percentage = linesRead / totalLines
                    if (percentage % 5 == 0):
                        print("Lines read: ", percentage)
                    fenString = line.split(",")[0]
                    value = float(line.split(",")[1])
                    board = chess.Board()
                    board.set_board_fen(fenString.split(" ")[0])
                    yield board, value
                except IndexError:
                    continue

    def getDataLoader(self, batchSize: int, shuffle: bool = False) -> ChessDataLoader:
        return ChessDataLoader(self, batch_size=batchSize)

    def evaluateUsingStockFish(board: chess.Board, engine: chess.engine.SimpleEngine | None = None) -> float:
        if engine is None:
            with chess.engine.SimpleEngine.popen_uci("project/chess_engines/stockfish/stockfish-windows-x86-64-avx2.exe") as new_engine:
                info = new_engine.analyse(board, limit=chess.engine.Limit(time=0.2, depth=4))
                pawnScore = info["score"].white().score(mate_score=100000) / 100.0
                return pawnScore
        info = engine.analyse(board, limit=chess.engine.Limit(time=0.1, depth=3))
        score = info["score"].white()
        if score.is_mate():
            # omit mates, network need not be training to evaluate mate score for it can be easily found using code.
            return None
        pawnScore = score.score() / 100.0
        return pawnScore