import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from functools import reduce
from operator import add
from tqdm import tqdm

from .generic_extractor import GenericFeatureExtractor

RF = 291.0
STRIDE = 16.0
PADDING = 145.0

# Stage depths for ImageNet models
_IN_STAGE_DS = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}
TRANS_FUN = "bottleneck_transform"
NUM_GROUPS = 1
WIDTH_PER_GROUP = 64
STRIDE_1X1 = False
BN_EPS = 1e-5
BN_MOM = 0.1
RELU_INPLACE = True


def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        "basic_transform": BasicTransform,
        "bottleneck_transform": BottleneckTransform,
    }
    err_str = "Transformation function '{}' not supported"
    assert name in trans_funs.keys(), err_str.format(name)
    return trans_funs[name]


class rgem(nn.Module):
    """Reranking with maximum descriptors aggregation"""

    def __init__(self, pr=2.5, size=5):
        super(rgem, self).__init__()
        self.pr = pr
        self.size = size
        self.lppool = nn.LPPool2d(self.pr, int(self.size), stride=1)
        self.pad = nn.ReflectionPad2d(int((self.size - 1) // 2.0))

    def forward(self, x):
        nominater = (self.size**2) ** (1.0 / self.pr)
        x = 0.5 * self.lppool(self.pad(x / nominater)) + 0.5 * x
        return x


class gemp(nn.Module):
    """Reranking with maximum descriptors aggregation"""

    def __init__(self, p=4.4, eps=1e-8):
        super(gemp, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        x = x.clamp(self.eps).pow(self.p)
        x = torch.nn.functional.adaptive_avg_pool1d(x, 1).pow(1.0 / (self.p))
        return x


class relup(nn.Module):
    """Reranking with maximum descriptors aggregation"""

    def __init__(self, alpha=0.014):
        super(relup, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        x = x.clamp(self.alpha)
        return x


class GlobalHead(nn.Module):
    def __init__(self, w_in, nc, pp=3):
        super(GlobalHead, self).__init__()
        self.pool = GeneralizedMeanPoolingP(norm=pp)
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(
            1.0 / self.p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + str(self.p)
            + ", "
            + "output_size="
            + str(self.output_size)
            + ")"
        )


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """Same, but norm is trainable"""

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class BasicTransform(nn.Module):
    """Basic transformation: 3x3, BN, ReLU, 3x3, BN."""

    def __init__(self, w_in, w_out, stride, w_b=None, num_gs=1):
        err_str = "Basic transform does not support w_b and num_gs options"
        assert w_b is None and num_gs == 1, err_str
        super(BasicTransform, self).__init__()
        self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False)
        self.a_bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.a_relu = nn.ReLU(inplace=RELU_INPLACE)
        self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, BN, ReLU, 3x3, BN, ReLU, 1x1, BN."""

    def __init__(self, w_in, w_out, stride, w_b, num_gs, relup_):
        super(BottleneckTransform, self).__init__()
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3
        (s1, s3) = (stride, 1) if STRIDE_1X1 else (1, stride)
        self.a = nn.Conv2d(w_in, w_b, 1, stride=s1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b, eps=BN_EPS, momentum=BN_MOM)
        if w_out == 256 and relup_:
            self.a_relu = relup(0.014)
        else:
            self.a_relu = relup(0.0)
        self.b = nn.Conv2d(w_b, w_b, 3, stride=s3, padding=1, groups=num_gs, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b, eps=BN_EPS, momentum=BN_MOM)
        if w_out == 256 and relup_:
            self.b_relu = relup(0.014)
        else:
            self.b_relu = relup(0.0)
        self.c = nn.Conv2d(w_b, w_out, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBlock(nn.Module):
    """Residual block: x + F(x)."""

    def __init__(
        self, w_in, w_out, stride, trans_fun, w_b=None, num_gs=1, relup_=False
    ):
        super(ResBlock, self).__init__()
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.f = trans_fun(w_in, w_out, stride, w_b, num_gs, relup_)
        self.w_in = w_in
        self.w_out = w_out
        if relup_:
            self.relup = relup(0.014)
        else:
            self.relup = relup(0.0)
        self.relu = relup(0.0)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        if self.w_out != 2048:
            x = self.relup(x)
        else:
            x = self.relu(x)
        return x


class ResStage(nn.Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b=None, num_gs=1, relup_=False):
        super(ResStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_fun = get_trans_fun(TRANS_FUN)
            res_block = ResBlock(
                b_w_in, w_out, b_stride, trans_fun, w_b, num_gs, relup_
            )
            self.add_module("b{}".format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class ResStage_basetransform(nn.Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b=None, num_gs=1):
        super(ResStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_fun = "basic_transform"
            res_block = ResBlock(b_w_in, w_out, b_stride, trans_fun, w_b, num_gs)
            self.add_module("b{}".format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.relu = nn.ReLU(RELU_INPLACE)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResNet(nn.Module):
    """ResNet model."""

    def __init__(self, RESNET_DEPTH, REDUCTION_DIM, relup_):
        super(ResNet, self).__init__()
        self.RESNET_DEPTH = RESNET_DEPTH
        self.REDUCTION_DIM = REDUCTION_DIM
        self.RELU_P = relup_
        self._construct()

    def _construct(self):
        g, gw = NUM_GROUPS, WIDTH_PER_GROUP
        (d1, d2, d3, d4) = _IN_STAGE_DS[self.RESNET_DEPTH]
        w_b = gw * g
        self.stem = ResStemIN(3, 64)
        self.s1 = ResStage(
            64, 256, stride=1, d=d1, w_b=w_b, num_gs=g, relup_=self.RELU_P
        )
        self.s2 = ResStage(
            256, 512, stride=2, d=d2, w_b=w_b * 2, num_gs=g, relup_=self.RELU_P
        )
        self.s3 = ResStage(
            512, 1024, stride=2, d=d3, w_b=w_b * 4, num_gs=g, relup_=self.RELU_P
        )
        self.s4 = ResStage(
            1024, 2048, stride=2, d=d4, w_b=w_b * 8, num_gs=g, relup_=self.RELU_P
        )
        self.head = GlobalHead(2048, nc=self.REDUCTION_DIM)

        # SuperGlobal stuff
        self.pool = nn.MaxPool2d(3, 1, padding=1)
        self.rgem = rgem()
        self.gemp = gemp()
        # self.sgem = sgem() # skipped scale gem

    def forward(self, x, gemp=False, rgem=False):
        x = self.stem(x)
        x1 = self.s1(x)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        x4 = self.s4(x3)
        if rgem and x4.shape[2] >= 5 and x4.shape[3] >= 5:
            x5 = self.rgem(x4)
        else:
            x5 = x4
        if gemp:
            x5 = x5.view(x5.shape[0], x5.shape[1], -1)
            x6 = self.gemp(x5)
        else:
            x6 = self.head.pool(x5)
        x6 = x6.view(x6.size(0), -1)
        if gemp or rgem:
            # this is in the SG but not in CVNet, never mentioned in the paper, and plays a role in the performance
            x6 = F.normalize(x6, p=2, dim=-1)
        x = self.head.fc(x6)
        return x, x3

    def forward_local(self, x):
        x = self.stem(x)
        x1 = self.s1(x)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        return x3


# Original code: HSNet (https://github.com/juhongm999/hsnet)
def extract_feat_res_pycls(img, backbone):
    nbottlenecks = [3, 4, 23, 3]
    feat_ids = [30]
    bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
    lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])

    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.stem(img)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = (
            backbone.__getattr__("s%d" % lid)
            .__getattr__("b%d" % (bid + 1))
            .f.forward(feat)
        )

        if bid == 0:
            res = (
                backbone.__getattr__("s%d" % lid)
                .__getattr__("b%d" % (bid + 1))
                .proj.forward(res)
            )
            res = (
                backbone.__getattr__("s%d" % lid)
                .__getattr__("b%d" % (bid + 1))
                .bn.forward(res)
            )
        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = (
            backbone.__getattr__("s%d" % lid)
            .__getattr__("b%d" % (bid + 1))
            .relu.forward(feat)
        )

    return feats


def generate_coordinates(h, w):
    """generate coorinates
    Returns: [h*w, 2] FloatTensor
    """
    x = torch.floor(torch.arange(0, float(w * h)) / w)
    y = torch.arange(0, float(w)).repeat(h)

    coord = torch.stack([x, y], dim=1)
    return coord


def calculate_receptive_boxes(height, width, rf, stride, padding):
    coordinates = generate_coordinates(height, width)
    point_boxes = torch.cat([coordinates, coordinates], dim=1)
    bias = torch.FloatTensor([-padding, -padding, -padding + rf - 1, -padding + rf - 1])
    rf_boxes = stride * point_boxes + bias
    return rf_boxes


def non_maxima_suppression_2d(heatmap):
    hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    keep = heatmap == hmax
    return keep


def calculate_keypoint_centers(rf_boxes):
    """compute feature centers, from receptive field boxes (rf_boxes).
    Args:
        rf_boxes: [N, 4] FloatTensor.
    Returns:
        centers: [N, 2] FloatTensor.
    """
    xymin = torch.index_select(rf_boxes, dim=1, index=torch.LongTensor([0, 1]).cuda())
    xymax = torch.index_select(rf_boxes, dim=1, index=torch.LongTensor([2, 3]).cuda())
    return (xymax + xymin) / 2.0


def load_cvnet(model_name, relup_=False):
    if "resnet50" in model_name:
        model = ResNet(50, 2048, relup)
    else:
        model = ResNet(101, 2048, relup)

    weight_path = f"feature_extraction/pretrained_models/cvnet/{model_name}.pt"
    weight = torch.load(weight_path)
    weight_new = {}
    for i, j in zip(weight["model_state"].keys(), weight["model_state"].values()):
        weight_new[i.replace("encoder_q.", "")] = j

    mis_key = model.load_state_dict(weight_new, strict=False)
    return model


class CVNetModel(GenericFeatureExtractor):
    def __init__(self, **kargs):
        super(CVNetModel, self).__init__(**kargs)
        assert (
            self.pooling == "head"
        ), f"{self.pooling.upper()} pooling operation is not appicable for CVNet models"

        if "superglobal" in self.name:
            self.model = load_cvnet(
                self.name.replace("superglobal", "cvnet"), relup_=True
            )
        else:
            self.model = load_cvnet(self.name)

    def _extract_features(self, x):
        x = torch.flip(x, (1,))
        if "superglobal" in self.name:
            return self.model(x, True, True)[0]
        else:
            return self.model(x)[0]
