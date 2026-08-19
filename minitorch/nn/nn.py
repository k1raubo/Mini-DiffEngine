from minitorch import scalar
import minitorch.nn.functional as F
import random


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):
        if isinstance(value, scalar):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value

        super().__setattr__(name, value)

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Cell(Module):
    def __init__(self):
        super().__init__()
        self.w = scalar(random.uniform(-1, 1), requires_grad=True)
        self.b = scalar(random.uniform(-1, 1), requires_grad=True)

    def forward(self, x):
        return self.w * x + self.b


class Perceptron(Module):
    def __init__(self):
        super().__init__()
        self.cell = Cell()

    def forward(self, x):
        return F.sigmoid(self.cell(x))