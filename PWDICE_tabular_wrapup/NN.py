import torch
import torch.nn as nn

class Onehot_Predictor(nn.Module):
    def __init__(self, input_size, use_bn=False):
        super().__init__()
        middle_size = 256
        if not use_bn:
            self.net = nn.Sequential(
                nn.Linear(input_size, middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, middle_size),
                nn.ReLU(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_size, middle_size),
                nn.BatchNorm1d(middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, middle_size),
                nn.BatchNorm1d(middle_size),
                nn.ReLU(),
            )
        self.net_a = nn.Linear(middle_size, 1)
        self.net_b = nn.Linear(middle_size, 1)
        self.net_c = nn.Linear(middle_size, 1)
        # self.net_a, self.net_b, self.net_c = nn.Linear(input_size, 1, bias=False),  nn.Linear(input_size, 1, bias=False), nn.Linear(input_size, 1, bias=False)
        self.net_d = nn.parameter.Parameter(torch.rand(1, requires_grad=True), requires_grad=True)
        
    def forward(self, x):
        v = self.net(x)
        return self.net_a(v), self.net_b(v), self.net_c(v), self.net_d
        
class TwoDimState_Predictor(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
        middle_size = 256
        if use_bn:
            self.net = nn.Sequential(
                nn.Linear(2, middle_size),
                nn.BatchNorm1d(middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, middle_size),
                nn.BatchNorm1d(middle_size),
                nn.ReLU(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(2, middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, middle_size),
                nn.ReLU(),
            )
        self.net_a = nn.Linear(middle_size, 1)
        self.net_b = nn.Linear(middle_size, 1)
        self.net_c = nn.Linear(middle_size, 1)
        self.net_d = nn.parameter.Parameter(torch.rand(1, requires_grad=True), requires_grad=True)
    def forward(self, x):
        v = self.net(x)
        return self.net_a(v), self.net_b(v), self.net_c(v), self.net_d


class Onehot_Predictor_Independent(nn.Module):
    def __init__(self, input_size, use_bn=False):
        super().__init__()
        middle_size = 256
        
        self.net_a = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, 1),
        )
        self.net_b = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, 1),
        )
        self.net_c = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, 1),
        )
        
        # self.net_a, self.net_b, self.net_c = nn.Linear(input_size, 1, bias=False),  nn.Linear(input_size, 1, bias=False), nn.Linear(input_size, 1, bias=False)
        self.net_d = nn.parameter.Parameter(torch.rand(1, requires_grad=True), requires_grad=True)
        
    def forward(self, v):
        return self.net_a(v), self.net_b(v), self.net_c(v), self.net_d

class TwoDimState_Linear_Predictor_Independent(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
        middle_size = 256
        
        self.net_a = nn.Sequential(
            nn.Linear(2, middle_size),
            nn.Linear(middle_size, 1)
        )
        self.net_b = nn.Sequential(
            nn.Linear(2, middle_size),
            nn.Linear(middle_size, 1)
        )
        self.net_c = nn.Sequential(
            nn.Linear(2, middle_size),
            nn.Linear(middle_size, 1)
        )
        self.net_d = nn.parameter.Parameter(torch.rand(1, requires_grad=True), requires_grad=True)
        
    def forward(self, v):
        return self.net_a(v), self.net_b(v), self.net_c(v), self.net_d

class TwoDimState_Predictor_Independent(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
        middle_size = 256
        
        self.net_a = nn.Sequential(
            nn.Linear(2, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, 1)
        )
        self.net_b = nn.Sequential(
            nn.Linear(2, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, 1)
        )
        self.net_c = nn.Sequential(
            nn.Linear(2, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, 1)
        )
        self.net_d = nn.parameter.Parameter(torch.rand(1, requires_grad=True), requires_grad=True)
        
    def forward(self, v):
        return self.net_a(v), self.net_b(v), self.net_c(v), self.net_d
