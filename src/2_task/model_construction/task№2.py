import torch


def target_function(x):
    return 2**x * torch.sin(2**-x)


class RegressionNet(torch.nn.Module):
    def __init__(self, n_hidden_neu):
        super(RegressionNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neu)
        self.act1 = torch.nn.Sigmoid()  # Лучше взять tanh функцию активации
        self.fc2 = torch.nn.Linear(n_hidden_neu, n_hidden_neu)
        self.act2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(n_hidden_neu, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


net = RegressionNet(30)  # 30 - кол-во нейронов


x_train = torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)


optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


def loss(pred, target):
    loss = (pred - target)**2
    return loss.mean()


for epoch_index in range(200):
    optimizer.zero_grad()

    y_pred = net.forward(x_train)
    loss_value = loss(y_pred, y_train)
    loss_value.backward()
    optimizer.step()


def metric(pred, target):
    return (pred - target).abs().mean()


print(metric(net.forward(x_validation), y_validation).item())
