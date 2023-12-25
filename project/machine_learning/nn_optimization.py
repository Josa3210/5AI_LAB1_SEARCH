import os
from typing import Type
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import torch.optim
import torch.cuda
from bayes_opt import BayesianOptimization
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from project.machine_learning.parsing import ChessDataLoader, DataParser
from project.machine_learning.neural_network_heuristic import CanCaptureHeuristic, NeuralNetworkHeuristic
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
    print(" " * 200, end='\r')


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
            print("Validating: {", "=" * percentage, " " * (100 - percentage), "} ", round(percentage), "%", end='\r')

    print(" " * 200, end='\r')
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

    def run(nL1: int, nL2: int) -> float:

        nL1: int = int(nL1)
        nL2: int = int(nL2)
        nL3: int = 0
        lr: float = 0.02815
        wd: float = 0.01
        do: float = 0.3
        epochs: int = 5

        model = CanCaptureHeuristic(nL1=nL1, nL2=nL2, nL3=nL3, activation=nn.ReLU(), dropoutRate=do)
        if torch.cuda.is_available():
            model.to(device='cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
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

        train(model=model, optimizer=optimizer, criterion=criterion, numberOfEpochs=epochs, dataLoader=trainDataLoader)
        loss = evaluate(model=model, criterion=criterion, testDataLoader=testDataLoader)
        return -loss


if __name__ == "__main__":
    # First, optimize the learning rate --> lr = 0.04322, 0.02815
    pboundslr = {'lr': (10e-5, 0.1)}

    # Second, optimize the amount of nodes and amount of layers
    # 1 layer: --> not much success...  60 < nL1 < 75 or nL1 = 112
    pboundsnL1 = {'nL1': (16, 256)}
    # 2 layers:
    pboundsnL2 = {'nL1': (16, 256), 'nL2': (16, 256)}
    # 3 layers:
    pboundsnL3 = {'nL1': (16, 256), 'nL2': (16, 256), 'nL3': (16, 256)}

    # Third, optimize batch size
    pboundsBatch = {'batch': (16, 64)}

    # Fourth, optimize activation function
    # Relu -->
    # LeakyRelu -->

    # Fifth, optimize dropout and weight_decay
    pboundsnWdDo = {'wd': (0, 0.1), 'do': (0, 0.6)}

    # Sixth, optimize number of epochs
    pboundsEpochs = {'ep': (1, 20)}

    # Last optimization with best parameters
    # pbounds = {'nL1': (64, 2048), 'nL2': (64, 2048), 'nL3': (64, 2048), 'lr': (0.0001, 0.0005), 'wd': (0, 0.5)}
    optimizer = BayesianOptimization(f=Objective.run, pbounds=pboundsnL2)

    optimizer.maximize(5, 15)
    maxVals = optimizer.max

    with open("D:\\_Opslag\\GitKraken\\5AI_LAB1_SEARCH\\project\\data\\params.txt", "a") as f:
        f.write("\nNew optimization")
        f.write("-"*100)
        f.write(optimizer.res)
        f.write("\nOptimal values:" + maxVals)
        f.write("-" * 100)

    print(maxVals)
