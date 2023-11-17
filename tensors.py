import torch

from options.options import Options
from utilities.utils import plot_tensor, mse, init_pytorch, not_implemented


def create_image(options: Options) -> torch.Tensor:
    """
    TODO: implement this method
    use options to put the tensor to the correct device.
    """
    return not_implemented()


def lin_layer_forward(weights: torch.Tensor, random_image: torch.Tensor) -> torch.Tensor:
    """TODO: implement this method"""
    return not_implemented()


def tensor_network():
    target = torch.FloatTensor([0.5]).to(options.device)
    print(f"The target is: {target.item():.2f}")
    plot_tensor(target, "Target")

    input_tensor = torch.FloatTensor([0.4, 0.8, 0.5, 0.3]).to(options.device)
    weights = torch.FloatTensor([0.1, -0.5, 0.9, -1]).to(options.device)
    """START TODO:  ensure that the tensor 'weights' saves the computational graph and the gradients after backprop"""

    """END TODO"""

    # remember the activation a of a unit is calculated as follows:
    #      T
    # a = W * x, with W the weights and x the inputs of that unit
    output = lin_layer_forward(weights, input_tensor).to(options.device)
    print(f"Output value: {output.item(): .2f}")
    plot_tensor(output.detach(), "Initial Output")

    # We want a measure of how close we are according to our target
    loss = mse(output, target)
    print(f"The initial loss is: {loss.item():.2f}\n")

    # Lets update the weights now using our loss..
    print(f"The current weights are: {weights}")

    """START TODO: the loss needs to be backpropagated"""

    """END TODO"""

    print(f"The gradients are: {weights.grad}")
    """START TODO: implement the update step with a learning rate of 0.5"""
    # use tensor operations, recall the following formula we've seen during class: x <- x - alpha * x'

    """END TODO"""
    print(f"The new weights are: {weights}\n")

    # What happens if we forward through our layer again?
    output = lin_layer_forward(weights, input_tensor)
    print(f"Output value: {output.item(): .2f}")
    plot_tensor(output.detach(), "Improved Output")


if __name__ == "__main__":
    options = Options()
    init_pytorch(options)
    tensor_network()
