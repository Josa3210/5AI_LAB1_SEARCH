import os
from math import floor
from typing import Any, Type

import torch as t
import torch.nn as nn
import torch.optim
from torch.optim import Adam, Optimizer
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from project.machine_learning.neural_network_heuristic import \
    NeuralNetworkHeuristic
from project.machine_learning.parsing import ChessDataLoader, DataParser
from project.machine_learning.neural_network_heuristic import SimpleNeuralNetworkHeuristic, CanCaptureNeuralNetworkHeuristic


class Criterion:
    def __call__(self, results: t.Tensor, targets: t.Tensor) -> t.Tensor:
        pass


def train(model: nn.Module, optimizer: Optimizer, criterion: Criterion, numberOfEpochs, dataLoader: DataLoader, testDataLoader: DataLoader) -> None:
    reportingPeriod = 1000
    for i in range(numberOfEpochs):
        print(f"Starting epoch {i + 1}:")
        print("=====================================================================")
        runningLoss = 0
        reportLoss = 0
        for j, data in enumerate(dataLoader):
            inputs, targets = data
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            sumLoss = loss.item()
            runningLoss += sumLoss
            reportLoss += sumLoss
            if (j + 1) % reportingPeriod == 0:
                batchLoss = reportLoss / reportingPeriod
                reportLoss = 0
                # print(f"Current running loss: {runningLoss}")
                print(
                    f"Average loss over last {reportingPeriod} batches: {round(batchLoss, 5)} ")
        print(f"Finished training for epoch {i + 1}")
        testLoss = 0
        for j, data in enumerate(testDataLoader):
            inputs, targets = data

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            testLoss += loss.item()
            if j % 200:
                percentage = floor(j / len(testDataLoader) * 100)
                print("Evaluating: {", "=" * percentage,
                      " " * (100 - percentage), "}", end='\r')
        print("Evaluation", " " * 110)
        print(
            f"Avg loss over the test data: {round(testLoss / len(testDataLoader), 5)}")
        print("=====================================================================\n")
        torch.save(model, "project/data/simpleModel.pth")


def collectData(folder_path: str, heuristic : Type[NeuralNetworkHeuristic]) -> ChessDataLoader:
    files = os.listdir(folder_path)
    dataParsers: [DataParser] = []
    for file in files:
        if not file.split(".")[-1] == "cache":
            dataParser = DataParser(filePath=folder_path + "/" + file)
            dataParser.parse()
            dataParsers.append(dataParser)
    return ChessDataLoader(data_parsers=dataParsers, heuristic= heuristic)


if __name__ == '__main__':
    # TODO use the dataset to train a NeuralNetworkHeuristic, afterwards save it.
    model = SimpleNeuralNetworkHeuristic()
    optimizer = Adam(model.parameters(), lr=0.00001)
    # optimizer = torch.optim.Adam()
    criterion: Criterion = nn.MSELoss()

    # Specify the folder path you want to get filepaths from
    trainingFolderPath = "project/data/raw/training"
    validationFolderPath = "project/data/raw/validation"

    print("Loading in all training data:")
    trainDataLoader = collectData(trainingFolderPath, model.__class__)
    print(f"Total amount of trainingdata batches: {len(trainDataLoader)}")
    print("Loading in test data:")
    testDataLoader = collectData(validationFolderPath, model.__class__)

    print("Start training: \n")
    train(model=model, optimizer=optimizer, criterion=criterion, numberOfEpochs=5,
          dataLoader=trainDataLoader, testDataLoader=testDataLoader)
    pass
