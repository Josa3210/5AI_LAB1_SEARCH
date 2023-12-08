from typing import Generator

import chess


class DataParser():
    """
    This class is the place to crunch large datasets into useable data for training.
    """
    def __init__(self, filePath : str) -> None:
        self.filePath = filePath
        
    def parse(self) -> None:
        """
        Call this method to parse the file and hold it in memory.
        """
        # TODO
        pass
    
    def values(self) -> iter[tuple[chess.Board, float]]:
        """
        This function generates an iterable: https://www.youtube.com/watch?v=HnggP09mKpM
        Use this in a for loop when training the NN.
        """
        # TODO pass
    
    