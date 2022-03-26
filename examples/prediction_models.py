import torch
import torch.nn as nn
import pytorch_lightning as pl

class AvgModel(nn.Module):
    def __init__(self):
        super(AvgModel, self).__init__()
    
    def forward(self, x):
        """Model gets average player score in the previous matches

        Args:
            x torch.tensor: Player form matrix of shape (N, D, L)

        Returns:
            predicted score torch.tensor: (N, 1)
        """
        return x[:,0,:].mean(dim=1).reshape((-1, ))

class PrevModel(nn.Module):
    def __init__(self):
        super(PrevModel, self).__init__()

    def forward(self, x):
        """Model gets predicts next score as previous score of player

        Args:
            x (torch.tensor): Player form matrix of shape (N, D, L)

        Returns:
            predicted score (torch.tensor): predicted score (N, 1)
        """
        return x[:,0,-1].reshape((-1, ))

class LinearModel(nn.Module):
    def __init__(self, window_size=4, num_features=5):
        super(LinearModel, self).__init__()
        self.dim = window_size * num_features
        self.fc1 = nn.Linear(self.dim, 1).double()
    
    def forward(self, x):
        x = x.reshape((x.shape[0], self.dim))
        return self.fc1(x).reshape((-1, ))

class RNNModel(nn.Module):
    def __init__(self, window_size=4,
                       num_features=5, num_layers=3,
                       hidden_dim=5):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(num_features, hidden_dim, num_layers).double()
        self.fc = nn.Linear(hidden_dim, 1).double()

    def forward(self, x):
        x = x.permute(2, 0, 1)
        h = self.rnn(x)
        o = self.fc(h[-1][-1])
        return o.reshape((-1, ))

class MagicAGIModel(nn.Module):
    def __init__(self, window_size=4, num_features=5) -> None:
        super(MagicAGIModel).__init__()
    
    def forward(self, x):
        pass

if __name__ == "__main__":
    input_tensor = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]]).reshape((2, 1, 4)).double()
    prev_model = PrevModel()
    avg_model = AvgModel()
    linear_model = LinearModel(num_features=1)
    rnn_model = RNNModel(num_features=1)
    print(avg_model.forward(input_tensor))
    print(prev_model.forward(input_tensor))
    print(linear_model.forward(input_tensor))
    print(rnn_model.forward(input_tensor).shape)
    rnn_model = RNNModel()