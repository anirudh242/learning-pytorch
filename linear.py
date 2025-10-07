import torch
from torch import nn

X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

model = nn.Linear(in_features=1, out_features=1)
print(f'Initial random weight: {model.weight.item():.3f}')
print(f'Initital random bias: {model.bias.item():.3f}')

loss_fn = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], loss: {loss.item():.4f}')

print(f'\n --- AFTER TRAINING ---')
print(f"Learned weight: {model.weight.item():.3f} (Should be close to 2.0)")
print(f"Learned bias: {model.bias.item():.3f} (Should be close to 1.0)")