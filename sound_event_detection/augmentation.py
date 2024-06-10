import torch
import torch.nn as nn

class time_shift(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            shift = torch.empty(1).normal_(self.mean, self.std).int().item()
            x = torch.roll(x, shift, dims=0)
        return x

class freq_shift(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            shift = torch.empty(1).normal_(self.mean, self.std).int().item()
            x = torch.roll(x, shift, dims=1)
        return x

class add_noise(nn.Module):
    def __init__(self, mean=0, std=1e-5):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        rand = torch.rand(x.shape)
        rand = torch.where(rand > 0.6, 1., 0.).to(x.device)
        white_noise = torch.normal(self.mean, self.std, size=x.shape, device=x.device)
        x = x + white_noise * rand  # add noise
        return x

class time_mask(nn.Module):
    def __init__(self, n=1, p=50):
        super().__init__()
        self.p = p
        self.n = n

    def forward(self, x):
        time, _ = x.shape
        if self.training:
            for i in range(self.n):
                t = torch.empty(1, dtype=int).random_(self.p).item()
                start = max(time - t, 1)
                t_ = torch.empty(1, dtype=int).random_(start).item()
                x[t_:t_+t, :] = 0
        return x

class freq_mask(nn.Module):
    def __init__(self, n=1, p=10):
        super().__init__()
        self.p = p
        self.n = n

    def forward(self, x):
        _, freq = x.shape
        if self.training:
            for i in range(self.n):
                f = torch.empty(1, dtype=int).random_(self.p).item()
                start = max(freq - f, 1)
                f_ = torch.empty(1, dtype=int).random_(start).item()
                x[:, f_:f_+f] = 0
        return x