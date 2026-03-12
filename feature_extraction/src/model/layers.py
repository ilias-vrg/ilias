import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention2d(nn.Module):
    """
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    """

    def __init__(self, in_c, bn_eps=1e-5, bn_mom=0.1, act_fn="relu"):
        super(SpatialAttention2d, self).__init__()

        self.conv1 = nn.Conv2d(in_c, in_c, 1, 1)
        self.bn = nn.BatchNorm2d(in_c, eps=bn_eps, momentum=bn_mom)
        if act_fn.lower() in ["relu"]:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ["leakyrelu", "leaky", "leaky_relu"]:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_c, 1, 1, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score
        """
        x = self.conv1(x)
        x = self.bn(x)

        feature_map_norm = F.normalize(x, p=2, dim=1, eps=1e-15)

        x = self.act1(x)
        x = self.conv2(x)

        att_score = self.softplus(x)
        return feature_map_norm, att_score

    def __repr__(self):
        return self.__class__.__name__


class BinarizationLayer(nn.Module):
    def __init__(
        self,
        file_name=None,
        dims=None,
        bits=None,
        sigma=1e-6,
        trainable=True,
    ):
        super(BinarizationLayer, self).__init__()
        self.bits = bits
        self.dims = dims
        self.sigma = sigma
        self.trainable = trainable
        self.W = None
        if file_name is not None:
            self.load(file_name)
        elif dims is not None:
            self.bits = bits if bits is not None else dims
            self.init_params()

    def save(self, file_name):
        np.savez_compressed(file_name, proj=self.W.detach().cpu().numpy())

    def load(self, file_name):
        white = np.load(file_name)
        proj = torch.from_numpy(white["proj"]).float()
        self.init_params(proj)

    def init_params(self, proj=None):
        if proj is None:
            proj = torch.randn(self.dims, self.bits)
        self.W = nn.Parameter(proj, requires_grad=self.trainable)
        self.dims, self.bits = self.W.shape

    @staticmethod
    def _itq_rotation(v, n_iter, bit):
        r = np.random.randn(bit, bit)
        u11, s2, v2 = np.linalg.svd(r)

        r = u11[:, :bit]

        for _ in range(n_iter):
            z = np.dot(v, r)
            ux = np.ones(z.shape) * (-1.0)
            ux[z >= 0] = 1
            c = np.dot(ux.transpose(), v)
            ub, sigma, ua = np.linalg.svd(c)
            r = np.dot(ua, ub.transpose())
        z = np.dot(v, r)
        b = np.ones(z.shape) * -1.0
        b[z >= 0] = 1
        return b, r

    def train_itq(self, trainset):
        c = np.cov(trainset.transpose())

        l, pc = np.linalg.eig(c)

        l_pc_ordered = sorted(
            zip(l, pc.transpose()), key=lambda _p: _p[0], reverse=True
        )
        pc_top = np.array([p[1] for p in l_pc_ordered[: self.bits]]).transpose()

        v = np.dot(trainset, pc_top)

        b, rotation = self._itq_rotation(v, 50, self.bits)

        proj = np.dot(pc_top, rotation)
        self.init_params(proj)
        return proj

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        x = torch.matmul(x, self.W)
        if self.training and self.trainable:
            x = torch.erf(x / np.sqrt(2 * self.sigma))
        else:
            x = torch.sign(x)
        return x

    def __repr__(
        self,
    ):
        return "{}(dims={}, bits={})".format(
            self.__class__.__name__, self.W.shape[0], self.W.shape[1]
        )
