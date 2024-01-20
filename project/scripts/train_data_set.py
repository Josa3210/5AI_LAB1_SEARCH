import argparse
import os
import time
from math import floor
from typing import Type
import keyboard
from time import sleep
from os import _exit

import numpy as np
from matplotlib import pyplot as pt
import torch as t
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.cuda.amp import autocast
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
import random

import matplotlib.pyplot as plt

from project.machine_learning.neural_network_heuristic import NeuralNetworkHeuristic, CanCaptureHeuristic, CanCaptureHeuristicBit, WorldViewHeuristic
from project.machine_learning.parsing import ChessDataLoader, DataParser

isPaused = False
def pausePoint():
    global isPaused
    while isPaused:
        sleep(2)

class Criterion:
    def __call__(self, results: t.Tensor, targets: t.Tensor) -> t.Tensor:
        pass


class EarlyStopper:
    def __init__(self, patience, delta):
        self.minLoss = float('inf')
        self.counter = 0
        self.patience = patience
        self.delta = delta

    def early_stop(self, loss: float):
        if loss < self.minLoss:
            self.counter = 0
            self.minLoss = loss
        elif loss > self.minLoss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(model: nn.Module, optimizer: Optimizer, criterion: Criterion, dataLoader: DataLoader) -> float:
    _runningLoss = 0
    _reportLoss = 0
    reportingPeriod = 1000
    batch = 0
    model.train()
    for j, data in enumerate(dataLoader):
        pausePoint()
        inputs, targets = data
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        sumLoss = loss.item()
        _runningLoss += sumLoss
        _reportLoss += sumLoss

        # Report progress
        if (j + 1) % floor(reportingPeriod / 20) == 0:
            percentage = floor((j % reportingPeriod) / reportingPeriod * 100)
            print("Training: {", "=" * percentage,
                  " " * (100 - percentage), "}", end='\r')

        # Report loss
        if (j + 1) % reportingPeriod == 0:
            batch += 1
            print(" " * 130, end="\r")
            batchLoss = _reportLoss / reportingPeriod
            _reportLoss = 0
            print(
                f"{batch}: Average loss over last {reportingPeriod} batches: {round(batchLoss, 5)} ")
    print(" " * 130, end="\r")
    _averageTrainingLoss = _runningLoss / (len(dataLoader)//batchSize)
    return round(_averageTrainingLoss, 5)


def validate(model: nn.Module, criterion: Criterion, validationDataLoader: DataLoader) -> float:
    _validationLoss = 0
    model.eval()
    with torch.no_grad():
        for j, data in enumerate(validationDataLoader):
            pausePoint()
            inputs, targets = data

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            _validationLoss += loss.item()
            if j % 20:
                percentage = floor(
                    j / (len(validationDataLoader)//batchSize) * 100)
                print("Evaluating: {", "=" * percentage,
                      " " * (100 - percentage), "}", end='\r')

    return round(_validationLoss / (len(validationDataLoader)//batchSize), 5)


def collectData(folder_path: str, heuristic: Type[NeuralNetworkHeuristic], batchSize: int) -> ChessDataLoader:
    files = os.listdir(folder_path)
    dataParsers: [DataParser] = []
    for file in files:
        pausePoint()
        if not file.split(".")[-1] == "cache":
            dataParser = DataParser(filePath=folder_path + "/" + file)
            dataParser.parse()
            dataParsers.append(dataParser)
    return ChessDataLoader(data_parsers=dataParsers, heuristic=heuristic, batch_size=batchSize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="DataTrainer", description="Train our model using datasets located in /project/data/raw")
    parser.add_argument('-l', '--learning-rate', default=0.0001, type=float)
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-e', '--epochs', default=5, type=int)
    parser.add_argument('--preload', default=None, type=str,
                        help="Continue training on a previous model, value is location to model")
    args = parser.parse_args()
    learningRate = args.learning_rate
    batchSize = args.batch_size
    numberOfEpochs = args.epochs
    preload = (args.preload)

    print(
        f"The learning parameters are:\n- Learning rate:\t{learningRate}\n- batchSize:\t\t{batchSize}\n- number of epochs:\t{numberOfEpochs}\n- preload:\t\t{preload}\n")

    # TODO use the dataset to train a NeuralNetworkHeuristic, afterwards save it.

    """
    vvv Insert model here vvv
    """
    # model: nn.Module = CanCaptureHeuristicBit(128, 64, 0, nn.LeakyReLU(), 0.3)
    model: NeuralNetworkHeuristic = WorldViewHeuristic()
    earlyStopper: EarlyStopper = EarlyStopper(1, 0.005)
    if preload is not None:
        model = torch.load(preload)
        model.train()
    if t.cuda.is_available():
        # print("Cuda was available, transferring data to GPU")
        model.to(device='cuda')

    optimizer = Adam(model.parameters(), lr=learningRate)
    criterion: Criterion = nn.MSELoss()

    # Implement pause at any moment loop
    
    def on_press_space(e):
        global isPaused, model
        if isPaused:
            return
        isPaused = True
        res = input(
            "You paused the program, do you wish to stop and save? [y/n]: ")
        if "y" in res:
            res = input("Under what name do you want to save the model?: ")
            saveLocation = f"project/data/models/{res}"
            torch.save(model, saveLocation)
            print(f"Saved the current model under {saveLocation}")
            _exit(0)
        else:
            res = input("Do you wish to continue? [y/n]: ")
            while res != "y":
                res = input("Do you wish to continue? [y/n]: ")
            isPaused = False
    keyboard.on_press_key(key='space', callback=on_press_space)
    
    print("If you want to pause training at any moment press SPACE")

    # Specify the folder path you want to get filepaths from
    trainingFolderPath = "project/data/raw/training"
    validationFolderPath = "project/data/raw/validation"

    trainDataLoader = collectData(
        trainingFolderPath, model.__class__, batchSize)
    validationDataLoader = collectData(
        validationFolderPath, model.__class__, batchSize)
    print(
        f"Total amount of batches:\n- training:\t{len(trainDataLoader)} datapoints -> {len(trainDataLoader)//batchSize} batches\n- validation:\t{len(validationDataLoader)} datapoints -> {len(validationDataLoader)//batchSize} batches\n")


    print("Start training: \n" + "=" * 100, '\n')
    startTime = time.perf_counter()

    trainingLossValues = []
    validationLossValues = []

    for i in range(numberOfEpochs):
        pausePoint()
        print(f"Starting epoch {i + 1}:\n" + "-" * 100)
        averageTrainingLoss = train(model=model, optimizer=optimizer,
                                    criterion=criterion, dataLoader=trainDataLoader)
        print(
            f"Finished training for epoch {i + 1} with average training loss: {averageTrainingLoss}")
        trainingLossValues.append(averageTrainingLoss)

        averageValidationLoss = validate(
            model=model, validationDataLoader=validationDataLoader, criterion=criterion)
        validationLossValues.append(averageValidationLoss)
        print(
            f"Finished validating for epoch {i + 1} with average validation loss: {averageValidationLoss}" + " " * 50 + "\n")
        torch.save(
            model, f"project/data/models/{model.getName()}_{i + 1}_0,{round(averageValidationLoss * 10000)}")
        if earlyStopper is not None and earlyStopper.early_stop(averageValidationLoss):
            print(
                f"Stopped early because previous loss {earlyStopper.minLoss} was lower than current loss {averageValidationLoss}")
            break

    endTime = time.perf_counter()
    seconds = endTime - startTime
    minutes = seconds // 60
    seconds = seconds - minutes * 60
    print(
        f"Time passed training: {round(minutes)} minutes {round(seconds)} seconds")

    # Plot the loss for every epoch
    pt.plot(trainingLossValues, 'r', label="training loss")
    pt.plot(validationLossValues, 'g', label="validation loss")
    pt.title("Loss over epochs")
    pt.legend(loc="upper right")

    pt.show()
    pass
