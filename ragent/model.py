import torch
import torch.nn.functional as F


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x