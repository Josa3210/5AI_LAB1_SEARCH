from typing import Any
import torch as t
import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader

from project.machine_learning.neural_network_heuristic import NeuralNetworkHeuristic
from project.machine_learning.parsing import DataParser
class Criterion:
    def __call__(self, results: t.Tensor, targets: t.Tensor) -> t.Tensor:
        pass

def train(model : nn.Module, optimizer : Optimizer, criterion : Criterion, numberOfEpochs, dataLoader : DataLoader, testDataLoader : DataLoader) -> None:
    reportingPeriod = 100
    for i in range(numberOfEpochs):
        print(f"Starting epoch {i +  1}:")
        runningLoss = 0
        for i, data in enumerate(dataLoader):
            inputs, targets = data
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            optimizer.step()
            
            sumLoss = sum(loss.tolist())
            runningLoss += sumLoss
            reportLoss += sumLoss
            if (i + 1) % reportingPeriod == 0:
                batchLoss = reportLoss / 100
                print(f"Current running loss: {runningLoss}")
                print(f"Average loss over last {reportingPeriod} batches is {batchLoss} ")
        print(f"Finished training for epoch {i + 1}")
        print("Starting evaluation")
        testLoss = 0
        for i, data in enumerate(testDataLoader):
            inputs, targets = data
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            testLoss += sum(loss.tolist())
        print(f"Avg loss over the training data: {testLoss/len(testDataLoader)}")
                    
            
            
            
    

if __name__ == '__main__' :
    # TODO use the dataset to train a NeuralNetworkHeuristic, afterwards save it.
    model = NeuralNetworkHeuristic()
    # optimizer = Adam(model.parameters(), lr= 0.0001)
    optimizer = None
    criterion : Criterion = nn.MSELoss()
    trainingData = DataParser(filePath="project/data/Carlsen.pgn")
    trainingData.parse()
    testData = DataParser(filePath="project/data/Carlsen.pgn")
    testData.parse()
    
    trainingDataLoader = trainingData.getDataLoader(32)
    testDataLoader = testData.getDataLoader(32)
    
    train(model=model, optimizer=optimizer, criterion=criterion, numberOfEpochs=5, dataLoader=trainingDataLoader, testDataLoader= testDataLoader)
    pass