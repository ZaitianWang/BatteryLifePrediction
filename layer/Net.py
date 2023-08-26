import random

import torch
from torch import nn
import math
import torch.nn.functional as F

MAX_INPUT_SIZE = 200


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=3000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class SimCLRModel(nn.Module):
    def __init__(self, base_model):
        super(SimCLRModel, self).__init__()
        self.base_model = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        features = self.base_model(x)
        embeddings = features.view(features.size(0), -1)
        projections = self.projection_head(embeddings)
        return embeddings, projections


class TransConv(nn.Module):
    def __init__(self, c_len):
        super(TransConv, self).__init__()
        self.feature_embed = 500
        self.cycle_feature_num = c_len
        self.d_model = 256
        self.dropout = 0.1

        self.pos = PositionalEncoding(self.d_model)
        self.linear = nn.Sequential(
            nn.Linear(128, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
        )
        self.embed = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
        )

        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, dropout=self.dropout,
                                                    batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)
        self.embed = nn.Linear(self.feature_embed, self.d_model)

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=5),
        )
        self.pooler = nn.Sequential(nn.Linear(self.cycle_feature_num, self.d_model),
                                    nn.ReLU(),
                                    nn.Linear(self.d_model, 1))

    def encode(self, x1):
        x1 = x1.permute(0, 2, 3, 1)  # [b,window,feature_embed,feature_num]
        x1 = self.pooler(x1)
        x1 = self.embed(x1.squeeze(-1))
        output = self.pos(x1)
        output = self.encoder(output).unsqueeze(1)
        output = self.features(output).flatten(1)
        return output

    def forward(self, x1):
        output = self.encode(x1)
        out = self.linear(output).squeeze()
        return out


def transform(batch, method):
    tensors = batch.clone()
    if method == "front_mask":
        for i in range(batch.shape[0]):
            idx = random.randint(0, 150)
            tensors[i, :, :idx, :] = 0
    if method == "random_mask":
        for i in range(batch.shape[0]):
            for _ in range(150):
                idx1 = random.randint(0, 6)
                idx2 = random.randint(0, 199)
                tensors[i, idx1, idx2, :] = 0
    return tensors


class JoinModel(nn.Module):
    def __init__(self,  feature_num):
        super(JoinModel, self).__init__()
        self.base_model = Trans(feature_num)
        self.mlp = nn.Sequential(
            nn.Linear(self.base_model.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        return self.base_model(x)
    
    def encode(self, x):
        return self.base_model.encode(x)

    def contrastive_loss(self, q_im, label):
        threshold = 40
        q_im = nn.functional.normalize(self.mlp(self.base_model.encode(q_im)), dim=1)
        label_diff = torch.abs(label - label[0]).float()
        weight = 1 + label_diff.unsqueeze(1)/2000
        q_im = q_im * weight
        arc = q_im[:1]
        neg = q_im[label_diff > threshold]
        pos = q_im[label_diff <= threshold][-1:]
        # TODO -1: the last one, change to the closest one
        # print(arc.shape, pos.shape, neg.shape)
        l_pos = torch.einsum('nc,nc->n', [arc, pos]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [arc, neg.T])
        logits = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        return F.cross_entropy(input=logits, target=labels)


class Trans(nn.Module):
    def __init__(self, feature_num):
        super(Trans, self).__init__()
        # self.feature_embed = 500
        self.feature_embed = 1024
        self.cycle_feature_num = feature_num # typically 5~10
        self.d_model = 512
        self.dropout = 0.1

        self.pos = PositionalEncoding(self.d_model)
        self.embed = nn.Sequential(
            nn.Linear(self.feature_embed, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, dropout=self.dropout,
                                                    batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)
        self.pooler = nn.Sequential(nn.Linear(self.cycle_feature_num, self.d_model),
                                    nn.ReLU(),
                                    nn.Linear(self.d_model, 1))
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, 2 * self.d_model),
            nn.ReLU(),
            nn.Linear(2 * self.d_model, self.d_model),
        )
        self.fc = nn.Linear(self.d_model, 1)

    def encode(self, x1, ):
        # [batch, 5, 100, 1024]
        x1 = x1.permute(0, 2, 3, 1)  # [b,window,feature_embed,feature_num]
        # [batch, 100, 1024, 5]
        x1 = self.pooler(x1)
        # [batch, 100, 1024, 1]
        x1 = self.embed(x1.squeeze(-1))
        # [batch, 100, 1024]
        # [batch, 100, 512]
        output = self.pos(x1)
        # [batch, 100, 512]
        output = torch.mean(self.encoder(output), dim=1)
        # [batch, 100, 512]
        # [batch, 512]
        return self.mlp(output)
        # [batch, 512]

    def forward(self, x1):
        # [batch, 5, 100, 1024]
        out = self.fc(self.encode(x1)).squeeze()
        # [batch, 512]
        # [batch, 1]
        # [batch]
        return out


class CNN(nn.Module):
    def __init__(self, feature_num):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(feature_num, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(32, 64, kernel_size=5),
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


# class CNN(nn.Module):
#     def __init__(self, feature_num):
#         super(CNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(feature_num, 16, kernel_size=(3, 5)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(3, 5), stride=(3, 5)),
#             nn.Conv2d(16, 32, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=3),
#             nn.Conv2d(32, 64, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=4, stride=4),
#         )
#
#         self.fc1 = nn.Sequential(nn.Linear(512, 256),
#                                  nn.ReLU(),
#                                  nn.Linear(256, 256),
#                                  )
#         self.fc2 = nn.Sequential(nn.Linear(256, 1))
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.shape[0], -1)
#         x = self.fc1(x).squeeze()
#         x = self.fc2(x).squeeze()
#         return x


class CNN3d(nn.Module):
    def __init__(self, c):
        super(CNN3d, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=3),
            nn.Conv3d(32, 64, kernel_size=(2, 3, 3)),
            nn.ReLU(),
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(32, 8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )
        self.fc = nn.Sequential(nn.Linear(272, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1),
                                )

    def forward(self, x):
        x = self.conv3d(x.unsqueeze(1))
        x = self.conv2d(x.squeeze())
        x = x.view(x.shape[0], -1)
        x = self.fc(x).squeeze()
        return x


class CNNv2(nn.Module):
    def __init__(self, c):
        super(CNNv2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )

        encoder_layers = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)
        self.fc = nn.Sequential(nn.Linear(32, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1),
                                )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        cls = torch.zeros_like(x[:, :1, :])
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)[:, 0, :]
        x = self.fc(x).squeeze()
        return x


class LSTM(nn.Module):
    def __init__(self, c):
        super(LSTM, self).__init__()
        input_size = 8
        input_dim = 500
        hidden_size = 64
        num_layers = 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.relu = nn.ReLU()
        self.pooler = nn.ModuleList(
            [nn.Linear(input_dim, hidden_size) for i in range(input_size)])
        self.lstm = nn.ModuleList(
            [nn.LSTM(hidden_size, hidden_size, batch_first=True) for i in range(input_size)])
        self.fc = nn.Sequential(nn.Linear(hidden_size * input_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, 1)
                                )

    def forward(self, x):
        lstm_out = []
        for i in range(len(self.lstm)):
            lstm_input = self.pooler[i](x[:, i, :])
            lstm_input = self.relu(lstm_input)
            _, (h, _) = self.lstm[i](lstm_input)
            h = h.view(x.shape[0], -1)
            lstm_out.append(h)
        lstm_out = torch.cat(lstm_out, dim=-1)
        out = self.fc(lstm_out)
        return out


class PatchEmbed(nn.Module):
    """
      Image to Patch Embedding
    """

    def __init__(self, img_size, patch_size, in_chans=7, embed_dim=700):
        super().__init__()

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        #
        # embed_dim表示切好的图片拉成一维向量后的特征长度
        #
        # 图像共切分为N = HW/P^2个patch块
        # 在实现上等同于对reshape后的patch序列进行一个PxP且stride为P的卷积操作
        # output = {[(n+2p-f)/s + 1]向下取整}^2
        # 即output = {[(n-P)/P + 1]向下取整}^2 = (n/P)^2
        #
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x  # x.shape is [8, 196, 768]


class ConvVIT(nn.Module):
    def __init__(self, c):
        super(ConvVIT, self).__init__()
        input_dim, hidden_dim, num_heads, num_layers = 768, 768, 6, 1
        self.patch_embedding = PatchEmbed((80, 500), (10, 10), c, input_dim)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, batch_first=True, norm_first=True), num_layers)
        self.pos = PositionalEncoding(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim * 2, hidden_dim))
        self.pre = nn.Linear(hidden_dim, 1)

    def forward(self, X):
        x = self.patch_embedding(X)
        x = self.embedding(x)
        cls = torch.zeros_like(x[:, :1, :])
        x = torch.cat([cls, x], dim=1)
        output = self.pos(x)
        output = self.encoder(output)
        cls_token = self.mlp(output)[:, 0, :]
        x = self.pre(cls_token).squeeze()
        return x


class ConvVITv2(nn.Module):
    def __init__(self, c):
        super(ConvVITv2, self).__init__()
        input_dim, hidden_dim, num_heads, num_layers = 768, 768, 6, 2
        self.patch_embedding = PatchEmbed((80, 500), (10, 10), c, input_dim)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, batch_first=True, norm_first=True), num_layers)
        self.pos = PositionalEncoding(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim * 2, hidden_dim))
        self.pre = nn.Linear(hidden_dim, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5),
            nn.Conv2d(32, 64, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=7, stride=7)
        )
        self.fc = nn.Sequential(nn.Linear(768, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1),
                                )

    def forward(self, X):
        x = self.patch_embedding(X)
        x = self.embedding(x)
        output = self.pos(x)
        output = self.encoder(output)
        output = self.mlp(output).unsqueeze(1)
        output = self.conv2(output)
        return self.fc(output).squeeze()
