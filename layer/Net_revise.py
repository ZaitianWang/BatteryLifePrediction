import torch
from torch import nn

from layer.Net import PositionalEncoding


class CNN(nn.Module):
    def __init__(self, c):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(128, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )
        self.fc = nn.Sequential(nn.Linear(2048, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1),
                                )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x).squeeze()
        return x


class Model(nn.Module):
    def __init__(self, s_len, c_len):
        super(Model, self).__init__()
        self.feature_embed = 500
        self.cycle_feature_num = c_len
        self.summary_feature_num = s_len
        self.out_dim = 256
        self.d_model = int(self.out_dim / 2)
        self.dropout = 0.1

        self.pos = PositionalEncoding(self.out_dim)
        self.linear = nn.Sequential(
            nn.Linear(self.out_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
        )
        self.embed1 = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

        encoder_layers = nn.TransformerEncoderLayer(d_model=self.out_dim, nhead=4, dropout=self.dropout,
                                                    batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.embed = nn.Linear(self.feature_embed, self.d_model)
        self.fc1 = nn.Sequential(nn.Linear(self.cycle_feature_num, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, 1))

        self.fc2 = nn.Sequential(nn.Linear(self.summary_feature_num, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model))

    def forward(self, x1, x2):
        x1 = x1.permute(0, 2, 3, 1)  # [b,window,feature_embed,feature_num]
        x1 = self.fc1(x1)
        x1 = self.embed(x1.squeeze())
        x2 = x2.permute(0, 2, 1)
        x2 = self.fc2(x2)
        x = torch.cat([x1, x2], dim=-1)
        x = self.embed1(x)
        cls = torch.zeros_like(x[:, :1, :])
        x = torch.cat([cls, x], dim=1)
        output = self.pos(x)
        output = self.encoder(output)[:, 0, :]
        out = self.linear(output).squeeze()
        return out


class Trans(nn.Module):
    def __init__(self, _, c_len):
        super(Trans, self).__init__()
        self.feature_embed = 500
        self.cycle_feature_num = c_len
        self.d_model = 256
        self.dropout = 0.1
        self.pos = PositionalEncoding(self.d_model)
        self.embed = nn.Sequential(nn.Linear(self.feature_embed, self.d_model))

        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, dropout=self.dropout,
                                                    batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.pooler = nn.Sequential(nn.Linear(self.cycle_feature_num, self.d_model),
                                    nn.ReLU(),
                                    nn.Linear(self.d_model, 1))
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.fc = nn.Linear(self.d_model, 1)

    def encode(self, x1):
        x1 = x1.permute(0, 2, 3, 1)  # [b,window,feature_embed,feature_num]
        x1 = self.pooler(x1)
        x1 = self.embed(x1.squeeze(-1))
        output = self.pos(x1)
        output = torch.mean(self.encoder(output), dim=1)
        return self.mlp(output)

    def forward(self, x1, _):
        out = self.fc(self.encode(x1)).squeeze()
        return out
