import torch
import torch.nn as nn

from options.classification_options import ClassificationOptions


class Print(nn.Module):
    """"
    This model is for debugging purposes (place it in nn.Sequential to see tensor dimensions).
    """

    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        return x


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        """START TODO: replace None with a Linear layer"""
        self.linear_layer = None
        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward the tensor x through the linear layer and return the outcome (replace None)"""
        x = None
        """END TODO"""
        return x

class Classifier(nn.Module):
    def __init__(self, options: ClassificationOptions):
        super().__init__()
        """ START TODO: fill in all three layers. 
            Remember that each layer should contain 2 parts, a linear layer and a nonlinear activation function.
            Use options.hidden_sizes to store all hidden sizes, (for simplicity, you might want to 
            include the input and output as well).
        """
        self.layer1 = nn.Sequential(

        )
        self.layer2 = nn.Sequential(

        )
        self.layer3 = nn.Sequential(

        )
        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward tensor x through all layers."""

        """END TODO"""
        return x


class ClassifierVariableLayers(nn.Module):
    def __init__(self, options: ClassificationOptions):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(options.hidden_sizes) - 1):
            self.layers.add_module(
                f"lin_layer_{i + 1}",
                nn.Linear(options.hidden_sizes[i], options.hidden_sizes[i + 1])
            )
            if i < len(options.hidden_sizes) - 2:
                self.layers.add_module(
                    f"relu_layer_{i + 1}",
                    nn.ReLU()
                )
            else:
                self.layers.add_module(
                    f"softmax_layer",
                    nn.Softmax(dim=1)
                )
        print(self)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        return x
