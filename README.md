# Mini-DiffEngine

Mini-DiffEngine is a small reverse mode automatic differentiation engine written from scratch in Python. It demonstrates how backpropagation and dynamic computational graph building work, using scalar values.

On top of the engine sits a tiny PyTorch style layer (`minitorch`) with modules, activation functions, a loss function and an SGD optimizer.

## 1. Project structure

- `engine/` - the autodiff core: `Variable`, `Function`, `Context`, topological ordering of the graph and gradient propagation.
- `minitorch/scalar.py` - the `Scalar` class (a scalar value with a gradient) and its operations: add, subtract, multiply, divide, power, negate, sigmoid, MSE.
- `minitorch/nn/` - `Module`, `Cell` (a single linear neuron) and `Perceptron` (a neuron with sigmoid).
- `minitorch/optim.py` - the `SGD` optimizer.
- `main.py` - an example training run of a simple perceptron on a small dataset.

## 2. How it works

Every value that should be tracked for gradients is wrapped in a `Scalar`. A `Scalar` is a `Variable`: it holds a `.data` value, an optional `.grad`, and a `requires_grad` flag.

When you do something like `a + b`, `a * b`, `sigmoid(x)` and so on, it does not just compute a number. Each of these operations is a `Function` subclass (`Add`, `Multiply`, `Sigmoid`, `MSELoss`, ...). Calling it goes through `Function.apply`, which:

1. Runs `forward` to compute the actual result.
2. Wraps that result in a new `Scalar`.
3. If any of the inputs require gradients, attaches a `grad_fn` to the output, remembering which `Function` produced it and which inputs (`parents`) it came from, plus anything the function saved for later (via `context.save_for_backward`).

This is what builds the graph: every new `Scalar` produced by an operation keeps a link back to the operation and inputs that created it, but only if gradients are needed.

Calling `.backward()` on the final result (usually a loss) does two things:

1. Walks the graph backward from that node to build a topological order of all the `Scalar`s that contributed to it.
2. Goes through that order in reverse, and for each node calls the `backward` method of the `Function` that produced it, passing in the gradient flowing from downstream. That gradient gets distributed to the node's parents and accumulated in their `.grad`.

So the forward pass builds the graph as a side effect of computing values, and the backward pass just replays that graph in reverse, applying the chain rule at each step. The `SGD` optimizer then reads `.grad` off each parameter and nudges `.data` in the opposite direction, scaled by the learning rate.

## 3. Requirements

Python 3.13 or newer. The project has no external dependencies.

## 4. Installation

```bash
git clone https://github.com/k1raubo/Mini-DiffEngine.git
cd Mini-DiffEngine
```

## 5. Example Usage

Run the example training script:

```bash
python main.py
```
