import os
import time
import argparse
from math import floor
from typing import Any, Type

import torch as t
import torch.nn as nn
import torch.optim
import torch.cuda
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from project.machine_learning.parsing import ChessDataLoader, DataParser
from project.machine_learning.neural_network_heuristic import SimpleNeuralNetworkHeuristic, CanCaptureNeuralNetworkHeuristic, NeuralNetworkHeuristic


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
            if (j + 1) % floor(reportingPeriod/20) == 0:
                percentage = floor((j % reportingPeriod) / reportingPeriod * 100)
                print("Evaluating: {", "=" * percentage,
                    " " * (100 - percentage), "}", end='\r')
            if (j + 1) % reportingPeriod == 0:
                print(" "*130, end="\r")
                batchLoss = reportLoss / reportingPeriod
                reportLoss = 0
                # print(f"Current running loss: {runningLoss}")
                print(
                    f"Average loss over last {reportingPeriod} batches: {round(batchLoss, 5)} ")
        print(" "*130, end="\r")
        print(f"Finished training for epoch {i + 1}")
        testLoss = 0
        with torch.no_grad():
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
        torch.save(model, f"project/data/models/{model.getName()}_temp")
    torch.save(model, f"project/data/models/{model.getName()}_{round(testLoss / len(testDataLoader), 5)}")


def collectData(folder_path: str, heuristic: Type[NeuralNetworkHeuristic], batchSize: int) -> ChessDataLoader:
    files = os.listdir(folder_path)
    dataParsers: [DataParser] = []
    for file in files:
        if not file.split(".")[-1] == "cache":
            dataParser = DataParser(filePath=folder_path + "/" + file)
            dataParser.parse()
            dataParsers.append(dataParser)
    return ChessDataLoader(data_parsers=dataParsers, heuristic=heuristic, batch_size= batchSize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="DataTrainer", description="Train our model using datasets located in /project/data/raw")
    parser.add_argument('-l', '--learning-rate', default=0.0001, type=float)
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-e', '--epochs', default=5, type=int)
    parser.add_argument('--preload', default=None, type=str, help="Continue training on a previous model, value is location to model")
    args = parser.parse_args()
    learningRate = (args.learning_rate)
    batchSize = (args.batch_size)    
    numberOfEpochs = (args.epochs)
    preload = args.preload
    print(f"The learning parameters are:\nLearning rate: {learningRate}\nbatchSize: {batchSize}\nnumber of epochs: {numberOfEpochs}\npreload: {preload}")
    # TODO use the dataset to train a NeuralNetworkHeuristic, afterwards save it.
    model : nn.Module = CanCaptureNeuralNetworkHeuristic()
    if preload is not None:
        model = torch.load(preload)
        model.train()
    if t.cuda.is_available():
        print("Cuda was available, transferring data to GPU")
        model.to(device='cuda')
    optimizer = Adam(model.parameters(), lr=learningRate)
    criterion: Criterion = nn.MSELoss()

    # Specify the folder path you want to get filepaths from
    trainingFolderPath = "project/data/raw/training"
    validationFolderPath = "project/data/raw/validation"

    print("Loading in all training data:")
    trainDataLoader = collectData(trainingFolderPath, model.__class__, batchSize)
    print(f"Total amount of trainingdata batches: {len(trainDataLoader)}\n")
    print("Loading in test data:")
    testDataLoader = collectData(validationFolderPath, model.__class__, batchSize)

    print("\nStart training: \n")
    startTime = time.perf_counter()
    train(model=model, optimizer=optimizer, criterion=criterion, numberOfEpochs=numberOfEpochs,
          dataLoader=trainDataLoader, testDataLoader=testDataLoader)
    endTime = time.perf_counter()
    seconds = endTime - startTime
    minutes = seconds / 60
    seconds = seconds - minutes*60
    print(f"Time passed training: {minutes} minutes {seconds} seconds")
    pass
