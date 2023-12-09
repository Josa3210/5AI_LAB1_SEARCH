from copy import deepcopy
from typing import Generator, Iterable

import chess
import chess.pgn
import chess.engine
import os

class DataParser():
    """
    This class is the place to crunch large datasets into useable data for training.
    """
    def __init__(self, filePath : str) -> None:
        self.filePath = filePath
        self.cachedFile = filePath + ".cache"
        
    def parse(self, overwriteCache = False) -> None:
        """
        Call this method to parse the file and hold it in memory.
        if overwriteCache is false it will look if the calculations were already done.
        """
        if not overwriteCache and os.path.exists(self.cachedFile):
            print(f"Found previous data, values are already available")
            return
        
        games : list[chess.pgn.Game] = []
        with open(self.filePath) as file:
            game = chess.pgn.read_game(file)
            while game is not None:
                games.append(game)
                game = chess.pgn.read_game(file)
        print(f"Data parser read in {len(games)} games.\nStart reading in boardStates")
        boards : list[chess.Board] = []
        for game in games:
            board : chess.Board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                boards.append(deepcopy(board))
        print(f"Read in {len(boards)} different positions.\nStarting stockfish evaluation")
        with open(self.cachedFile, 'w') as cacheFile:
            counter = 0
            for board in boards:
                fenString = board.fen()
                value = DataParser.evaluateUsingStockFish(board)
                cacheFile.write(f"{fenString},{value}\n")
                counter += 1
                if counter % 10 == 0:
                    print(f"Read in and evaluated {counter} out of {len(boards)} positions.")
                    cacheFile.flush()
        print("Evaluated all positions, data is now accessible via self.values")
        
    def values(self) -> Iterable[tuple[chess.Board, float]]:
        """
        This function generates an iterable: https://www.youtube.com/watch?v=HnggP09mKpM
        Use this in a for loop when training the NN.
        """
        with open(self.cachedFile) as data:
            line = data.readline()
            while line is not None:
                fenString = line.split(",")[0]
                value = float(line.split(",")[1])
                board = chess.Board()
                board.set_board_fen(fenString.split(" ")[0])
                yield (board, value)
                line = data.readline()
    
    def evaluateUsingStockFish(board: chess.Board) -> float:
        with chess.engine.SimpleEngine.popen_uci("project/chess_engines/stockfish/stockfish-windows-x86-64-avx2.exe") as engine:
            info = engine.analyse(board, limit=chess.engine.Limit(time=1.0, depth=4))
            return info["score"].white().score()/100.0
            
    