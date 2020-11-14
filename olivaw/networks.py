import torch
from torch import nn
from torch.nn import functional as F


class _AtariConvNet(nn.Module):
    def __init__(self, frame_size, nb_channels):
        super().__init__()
        opts = [8, 4, 3], [4, 2, 1], [0, 0, 0]
        w = self.out(frame_size[0], *opts)
        h = self.out(frame_size[1], *opts)

        self.convnet = nn.Sequential(
            nn.Conv2d(nb_channels, 32, kernel_size=8, stride=4),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(inplace=True)
        )
        self.out_dim = w * h * 64

    @staticmethod
    def out(w, kernel_sizes, strides, paddings):
        for k, s, p in zip(kernel_sizes, strides, paddings):
            w = (w - k + 2 * p) // s + 1
        return w

    def forward(self, x):
        return self.convnet(x)


class AtariDQN(nn.Module):
    def __init__(self, action_size, frame_size, stack_size, device):
        super().__init__()
        self.device = device

        self.convnet = _AtariConvNet(frame_size, stack_size)
        self.emb = nn.Linear(self.convnet.out_dim, 512)
        self.clf = nn.Sequential(
            nn.ELU(),
            nn.Linear(512, action_size)
        )

    def forward(self, stacked_frames):
        features = self.convnet(stacked_frames)
        emb = features.view(stacked_frames.shape[0], -1)
        emb = self.emb(emb)
        return {"qvalues": self.clf(emb), "emb": [emb]}


class AtariDuelingDQN(nn.Module):
    def __init__(self, action_size, frame_size, stack_size, device):
        super().__init__()
        self.device = device

        self.convnet = _AtariConvNet(frame_size, stack_size)
        self.state_emb = nn.Linear(self.convnet.out_dim, 256)
        self.state_stream = nn.Sequential(
            nn.ELU(),
            nn.Linear(256, 1)
        )
        self.adv_emb = nn.Linear(self.convnet.out_dim, 256)
        self.advantage_stream = nn.Sequential(
            nn.ELU(),
            nn.Linear(256, action_size)
        )

    def forward(self, stacked_frames):
        features = self.convnet(stacked_frames)
        emb = features.view(stacked_frames.shape[0], -1)

        emb_state = self.state_emb(emb)
        emb_adv = self.adv_emb(emb)

        v = self.state_stream(emb_state)
        a = self.advantage_stream(emb_adv)

        q = v + (a - a.mean(dim=1, keepdim=True))
        return {"qvalues": q, "emb": [emb_state, emb_adv]}
