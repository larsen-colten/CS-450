import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Reg(torch.nn.Module):
    def __init__(self):
        super(Reg, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(11, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.linear(x)
        return x


df = pd.read_csv('Reg\insurance.csv')
df = pd.get_dummies(df)

# normalization
targets = df['charges'].copy()

for i in df.columns:
    df[i] = (df[i] - min(df[i])) / (max(df[i] - min(df[i])))

# training data
X_train = torch.tensor(
    (df.drop('charges', axis=1).iloc[:1100]).to_numpy().astype(np.float32))
y_train = torch.tensor(
    (df['charges'].iloc[:1100]).to_numpy().astype(np.float32))

y_train = y_train.reshape(-1, 1)

# model

train_tensor = torch.utils.data.TensorDataset(X_train, y_train)
trainloader = torch.utils.data.DataLoader(train_tensor, batch_size=10)
model = Reg()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.01)

losses = []

epochs = 100
for e in range(epochs):
    running_loss = 0
    main_outs = []
    for data, targets in trainloader:

        optimizer.zero_grad()
        outs = model.forward(data)

        main_outs.extend(outs)
        loss = criterion(outs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
        losses.append(running_loss/len(trainloader))

plt.style.use('seaborn')
plt.plot(losses)
plt.show()

# testing
X_test = torch.tensor(
    (df.drop('charges', axis=1).iloc[1100:]).to_numpy().astype(pd.np.float32))
y_test = torch.tensor(
    (df['charges'].iloc[1100:]).to_numpy().astype(pd.np.float32))

y_test = y_test.reshape(-1, 1)

predictions = model(X_train) * (max(targets) - min(targets)) + min(targets)

# demoralize test predictions
predictions = model(X_test).clone().detach()

demoralize = predictions * (max(targets) - min(targets)) + min(targets)

sns.regplot(y_test.squeeze(), demoralize.squeeze())

plt.show()
