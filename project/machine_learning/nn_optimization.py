import os
from math import floor
from typing import Type

import torch.optim
import torch.cuda
from bayes_opt import BayesianOptimization
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from project.machine_learning.parsing import ChessDataLoader, DataParser
from project.machine_learning.neural_network_heuristic import CanCaptureNeuralNetworkHeuristic, NeuralNetworkHeuristic, NetworkToOptimize
import torch.nn as nn
import torch as t


class Criterion:
    def __call__(self, results: t.Tensor, targets: t.Tensor) -> t.Tensor:
        pass


def train(model: nn.Module, optimizer: Optimizer, criterion: Criterion, numberOfEpochs, dataLoader: DataLoader):
    for i in range(numberOfEpochs):
        runningLoss = 0
        for j, data in enumerate(dataLoader):
            inputs, targets = data
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            sumLoss = loss.item()
            runningLoss += sumLoss
            if j % 100 == 0:
                percentage = int(j / len(dataLoader) * 100)
                print("Training: {", "=" * percentage, " " * (100 - percentage), "} ", round(percentage), "%", end='\r')
    print(" " * 150, end='\r')


def evaluate(model: nn.Module, criterion: Criterion, testDataLoader: DataLoader) -> float:
    testLoss = 0
    for j, data in enumerate(testDataLoader):
        inputs, targets = data

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            testLoss += loss.item()

    if j % 200 == 0:
        percentage = int(j / len(testDataLoader) * 100)
        print("Training: {", "=" * percentage, " " * (100 - percentage), "} ", round(percentage), "%", end='\r')

    print(" " * 150, end='\r')
    averageLoss = round(testLoss / len(testDataLoader), 5)
    return averageLoss


def collectData(folder_path: str, heuristic: Type[NeuralNetworkHeuristic]) -> ChessDataLoader:
    files = os.listdir(folder_path)
    dataParsers: [DataParser] = []
    for file in files:
        if not file.split(".")[-1] == "cache":
            dataParser = DataParser(filePath=folder_path + "/" + file)
            dataParser.parse()
            dataParsers.append(dataParser)
    return ChessDataLoader(data_parsers=dataParsers, heuristic=heuristic)


class Objective:
    globalTrainDataLoader = None
    globalTestDataLoader = None

    def run(n_L1: int, n_L2: int, lr: int) -> float:

        n_L1: int = int(n_L1)
        n_L2: int = int(n_L2)
        lr: int = int(lr)

        model = NetworkToOptimize(n_L1=n_L1, n_L2=n_L2)
        if torch.cuda.is_available():
            model.to(device='cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion: Criterion = torch.nn.MSELoss()

        # Specify the folder path you want to get filepaths from
        trainingFolderPath = 'D:\\_Opslag\\GitKraken\\5AI_LAB1_SEARCH\\project\\data\\raw\\training'
        validationFolderPath = 'D:\\_Opslag\\GitKraken\\5AI_LAB1_SEARCH\\project\\data\\raw\\validation'

        if Objective.globalTrainDataLoader is None:
            trainDataLoader = collectData(trainingFolderPath, model.__class__)
            Objective.globalTrainDataLoader = trainDataLoader
        else:
            trainDataLoader = Objective.globalTrainDataLoader
        if Objective.globalTestDataLoader is None:
            testDataLoader = collectData(validationFolderPath, model.__class__)
            Objective.globalTestDataLoader = testDataLoader
        else:
            testDataLoader = Objective.globalTestDataLoader

        train(model=model, optimizer=optimizer, criterion=criterion, numberOfEpochs=3, dataLoader=trainDataLoader)
        loss = evaluate(model=model, criterion=criterion, testDataLoader=testDataLoader)
        return -loss


if __name__ == "__main__":
    pbounds = {'n_L1': (64, 512), 'n_L2': (8, 64), 'lr': (0.00002, 0.0004)}
    optimizer = BayesianOptimization(f=Objective.run, pbounds=pbounds)

    optimizer.maximize(3, 10)
    max = optimizer.max
    print(max)
