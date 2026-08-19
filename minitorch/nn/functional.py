from minitorch.scalar import *

def sigmoid(x):
    return Sigmoid.apply(x)


def mse_loss(prediction, target):
    return MSELoss.apply(prediction, target)

