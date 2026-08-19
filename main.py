import minitorch
import minitorch.nn as nn
import minitorch.nn.functional as F
import minitorch.optim as  optim


data = [
    (minitorch.scalar(-1), minitorch.scalar(0)),
    (minitorch.scalar(-0.5), minitorch.scalar(0)),
    (minitorch.scalar(0), minitorch.scalar(0)),
    (minitorch.scalar(0.5), minitorch.scalar(1)),
    (minitorch.scalar(1), minitorch.scalar(1)),
]

model = nn.Perceptron()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 8000

for epoch in range(epochs):
    total_loss = 0
    for x, target in data:

        y_pred = model(x)
        loss = F.mse_loss(y_pred, target)
        total_loss += loss.data

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

for x, target in data:
    y_pred = model(x)
    print(f"Input:  Pred: {y_pred.data:.4f} Target: {target.data}")