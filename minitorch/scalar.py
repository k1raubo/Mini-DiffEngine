from engine import Variable, Function
import math


class Scalar(Variable):
    def __init__(self, data, requires_grad=False):
        super().__init__(Scalar, requires_grad)
        self.data = float(data)

    def __add__(self, scalar):
        result = Add.apply(self, scalar)

        return result

    def __sub__(self, scalar):
        result = Subtract.apply(self, scalar)

        return result

    def __mul__(self, scalar):
        result = Multiply.apply(self, scalar)

        return result

    def __truediv__(self, scalar):
        result = Divide.apply(self, scalar)

        return result

    def __pow__(self, scalar):
        result = Power.apply(self, scalar)

        return result

    def __neg__(self):
        return Negate.apply(self)


class Add(Function):
    @staticmethod
    def forward(context, *args):
        return args[0].data + args[1].data

    @staticmethod
    def backward(context, grad_output):
        grad_x = grad_output * Scalar(1)
        grad_y = grad_output * Scalar(1)

        return grad_x, grad_y


class Subtract(Function):
    @staticmethod
    def forward(context, *args):
        return args[0].data - args[1].data

    @staticmethod
    def backward(context, grad_output):
        grad_x = grad_output * Scalar(1)
        grad_y = grad_output * Scalar(-1)

        return grad_x, grad_y


class Multiply(Function):
    @staticmethod
    def forward(context, *args):
        context.save_for_backward(*args)
        return args[0].data * args[1].data

    @staticmethod
    def backward(context, grad_output):
        x, y = context.saved_objects
        grad_x = grad_output * y
        grad_y = grad_output * x

        return grad_x, grad_y


class Divide(Function):
    @staticmethod
    def forward(context, *args):
        context.save_for_backward(*args)
        return args[0].data / args[1].data

    @staticmethod
    def backward(context, grad_output):
        x, y = context.saved_objects
        grad_x = grad_output / y
        grad_y = grad_output * (-x / y ** Scalar(2))

        return grad_x, grad_y


class Power(Function):
    @staticmethod
    def forward(context, *args):
        context.save_for_backward(*args)
        return args[0].data ** args[1].data

    @staticmethod
    def backward(context, grad_output):
        x, y = context.saved_objects
        grad_x = grad_output * y * x ** (y - Scalar(1))

        return grad_x


class Negate(Function):
    @staticmethod
    def forward(context, *args):
        return - args[0].data

    @staticmethod
    def backward(context, grad_output):
        return grad_output * Scalar(-1)


class Sigmoid(Function):
    @staticmethod
    def forward(context, *args):
        value = 1 / (1 + math.exp(-args[0].data))
        context.save_for_backward(Scalar(value))

        return value

    @staticmethod
    def backward(context, grad_output):
        (sigmoid_output,) = context.saved_objects
        return grad_output * sigmoid_output * (Scalar(1) - sigmoid_output)


class MSELoss(Function):
    @staticmethod
    def forward(context, *args):
        prediction = args[0]
        target = args[1]
        diff = prediction - target
        context.save_for_backward(diff)
        return diff.data ** 2

    @staticmethod
    def backward(context, grad_output):
        diff, = context.saved_objects
        grad_pred = grad_output * Scalar(2) * diff
        grad_target = None
        return grad_pred, grad_target