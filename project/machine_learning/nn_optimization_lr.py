import os
from math import floor
from typing import Type

import torch.optim
from bayes_opt import BayesianOptimization
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from project.machine_learning.parsing import ChessDataLoader, DataParser
from project.machine_learning.neural_network_heuristic import CanCaptureNeuralNetworkHeuristic, NeuralNetworkHeuristic
import torch.nn as nn
import torch as t


class Criterion:
    def __call__(self, results: t.Tensor, targets: t.Tensor) -> t.Tensor:
        pass


def train(model: nn.Module, optimizer: Optimizer, criterion: Criterion, numberOfEpochs, dataLoader: DataLoader):
    reportingPeriod = 1000
    averageLosses = []
    for i in range(numberOfEpochs):
        runningLoss = 0
        for j, data in enumerate(dataLoader):
            inputs, targets = data
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            sumLoss = loss.item()
            runningLoss += sumLoss
            if j % 500 == 0:
                percentage = (j / len(dataLoader))
                print("Training: {", "=" * percentage, " " * (100 - percentage), "}", end='\r')


def evaluate(model: nn.Module, criterion: Criterion, testDataLoader: DataLoader) -> float:
    testLoss = 0
    for j, data in enumerate(testDataLoader):
        inputs, targets = data

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        testLoss += loss.item()

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


def Objective(lr):
    model = CanCaptureNeuralNetworkHeuristic()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion: Criterion = torch.nn.MSELoss()

    # Specify the folder path you want to get filepaths from
    trainingFolderPath = "D:\\_Opslag\\GitKraken\\5AI_LAB1_SEARCH\\project\\data\\raw\\training"
    validationFolderPath = "D:\\_Opslag\\GitKraken\\5AI_LAB1_SEARCH\\project\\data\\raw\\validation"

    trainDataLoader = collectData(trainingFolderPath, model.__class__)
    testDataLoader = collectData(validationFolderPath, model.__class__)

    train(model=model, optimizer=optimizer, criterion=criterion, numberOfEpochs=5, dataLoader=trainDataLoader)
    loss = evaluate(model=model, criterion=criterion, testDataLoader=testDataLoader)
    return -loss


if __name__ == "__main__":
    pbounds = {'lr': (0.00002, 0.0004)}
    optimizer = BayesianOptimization(f=Objective, pbounds=pbounds)

    optimizer.maximize(3, 10)
