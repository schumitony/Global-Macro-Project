import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):

    def __init__(self, Imput_dim=10, nonlin=F.relu, dropout=0.5):

        super(Net, self).__init__()
        self.fc1 = nn.Linear(Imput_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
        # self.fc4 = nn.Linear(100, 10)
        # self.fc5 = nn.Linear(10, 1)

    def forward(self, x):
        # x = torch.from_numpy(x.values).float()
        # x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def loss(self, y_pred, y):
        return nn.MSELoss(y_pred, y)

    def optimizer(self, params, lr=0.1):
        return optim.SGD(params=params, lr=lr)
