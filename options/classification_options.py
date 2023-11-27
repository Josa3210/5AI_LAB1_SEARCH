from options.options import Options


class ClassificationOptions(Options):
    def __init__(self):
        super().__init__()
        # dataset related
        self.batch_size_test = 1000
        self.batch_size_train = 64

        # hyperparameters
        self.lr = 0.001
        self.num_epochs = 10
        self.hidden_sizes = [784, 784, 784, 10]
