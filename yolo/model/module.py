from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t

from yolo.config.config import AnchorConfig
from yolo.utils.bounding_box_utils import Vec2Box
from yolo.utils.logger import logger
from yolo.utils.module_utils import auto_pad, create_activation_function, round_up


# ----------- Basic Class ----------- #
class Conv(nn.Module):
    """A basic convolutional block that includes convolution, batch normalization, and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        *,
        activation: Optional[str] = "SiLU",
        **kwargs,
    ):
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2)
        self.act = create_activation_function(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class Pool(nn.Module):
    """A generic pooling block supporting 'max' and 'avg' pooling methods."""

    def __init__(self, method: str = "max", kernel_size: _size_2_t = 2, **kwargs):
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        pool_classes = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}
        self.pool = pool_classes[method.lower()](kernel_size=kernel_size, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)


class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)


# ----------- Detection Class ----------- #
class Detection(nn.Module):
    """A single YOLO Detection head for detection models"""

    def __init__(self, in_channels: Tuple[int], num_classes: int, *, reg_max: int = 16, use_group: bool = True,
                 is_exporting: bool = False, head_index: int = -1, stride: int = -1,
                 image_size: tuple[int, int] = (640, 640)):
        super().__init__()

        groups = 4 if use_group else 1
        anchor_channels = 4 * reg_max

        first_neck, in_channels = in_channels
        anchor_neck = max(round_up(first_neck // 4, groups), anchor_channels, reg_max)
        class_neck = max(first_neck, min(num_classes * 2, 128))

        self.anchor_conv = nn.Sequential(
            Conv(in_channels, anchor_neck, 3),
            Conv(anchor_neck, anchor_neck, 3, groups=groups),
            nn.Conv2d(anchor_neck, anchor_channels, 1, groups=groups),
        )
        self.class_conv = nn.Sequential(
            Conv(in_channels, class_neck, 3), Conv(class_neck, class_neck, 3), nn.Conv2d(class_neck, num_classes, 1)
        )

        self.anc2vec = Anchor2Vec(reg_max=reg_max)

        self.anchor_conv[-1].bias.data.fill_(1.0)
        self.class_conv[-1].bias.data.fill_(-10)  # TODO: math.log(5 * 4 ** idx / 80 ** 3)

        self.is_exporting = is_exporting
        if self.is_exporting:
            if head_index == -1:
                raise RuntimeError('Unable to determine head index')

            if stride > 0:
                head_stride = stride
            else:
                head_stride = 2 ** (head_index + 3)

            cfg = AnchorConfig([head_stride], None, None, None)
            self.converter = Vec2Box(None, cfg, image_size, 'cpu')

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor] | Tensor:
        anchor_x = self.anchor_conv(x)
        class_x = self.class_conv(x)
        anchor_x, vector_x = self.anc2vec(anchor_x)

        if self.is_exporting:
            pred_class, _, pred_bbox = self.converter([(class_x, anchor_x, vector_x)])
            pred_class = pred_class.sigmoid()

            output = torch.concat([pred_bbox, pred_class], axis=2)

            return output
        else:
            return class_x, anchor_x, vector_x


class IDetection(nn.Module):
    def __init__(self, in_channels: Tuple[int], num_classes: int, *args, anchor_num: int = 3, **kwargs):
        super().__init__()

        if isinstance(in_channels, tuple):
            in_channels = in_channels[1]

        out_channel = num_classes + 5
        out_channels = out_channel * anchor_num
        self.head_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.implicit_a = ImplicitA(in_channels)
        self.implicit_m = ImplicitM(out_channels)

    def forward(self, x):
        x = self.implicit_a(x)
        x = self.head_conv(x)
        x = self.implicit_m(x)

        return x


class MultiheadDetection(nn.Module):
    """Mutlihead Detection module for Dual detect or Triple detect"""

    def __init__(self, in_channels: List[int], num_classes: int, image_size: tuple[int, int] = (640, 640),
                 anchor_strides: Optional[List[int]] = None, **head_kwargs):
        super().__init__()
        DetectionHead = Detection

        if head_kwargs.pop("version", None) == "v7":
            DetectionHead = IDetection

        if anchor_strides is None:
            anchor_strides = [0] * len(in_channels)

        self.heads = nn.ModuleList(
            [DetectionHead((in_channels[0], in_channel), num_classes, head_index=i, stride=s, image_size=image_size, **head_kwargs) for i, (in_channel, s) in enumerate(zip(in_channels, anchor_strides))]
        )

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [head(x) for x, head in zip(x_list, self.heads)]


# ----------- Segmentation Class ----------- #
class Segmentation(nn.Module):
    def __init__(self, in_channels: Tuple[int], num_maskes: int):
        super().__init__()
        first_neck, in_channels = in_channels

        mask_neck = max(first_neck // 4, num_maskes)
        self.mask_conv = nn.Sequential(
            Conv(in_channels, mask_neck, 3), Conv(mask_neck, mask_neck, 3), nn.Conv2d(mask_neck, num_maskes, 1)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.mask_conv(x)
        return x


class MultiheadSegmentation(nn.Module):
    """Mutlihead Segmentation module for Dual segment or Triple segment"""

    def __init__(self, in_channels: List[int], num_classes: int, num_maskes: int, **head_kwargs):
        super().__init__()
        mask_channels, proto_channels = in_channels[:-1], in_channels[-1]

        self.detect = MultiheadDetection(mask_channels, num_classes, **head_kwargs)
        self.heads = nn.ModuleList(
            [Segmentation((in_channels[0], in_channel), num_maskes) for in_channel in mask_channels]
        )
        self.heads.append(Conv(proto_channels, num_maskes, 1))

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [head(x) for x, head in zip(x_list, self.heads)]


class Anchor2Vec(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        reverse_reg = torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1, 1)
        self.anc2vec = nn.Conv3d(in_channels=reg_max, out_channels=1, kernel_size=1, bias=False)
        self.anc2vec.weight = nn.Parameter(reverse_reg, requires_grad=False)

    def forward(self, anchor_x: Tensor) -> Tensor:
        anchor_x = rearrange(anchor_x, "B (P R) h w -> B R P h w", P=4)
        vector_x = anchor_x.softmax(dim=1)
        vector_x = self.anc2vec(vector_x)[:, 0]
        return anchor_x, vector_x


# ----------- Classification Class ----------- #
class Classification(nn.Module):
    def __init__(self, in_channel: int, num_classes: int, *, neck_channels=1024, **head_args):
        super().__init__()
        self.conv = Conv(in_channel, neck_channels, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(neck_channels, num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.pool(self.conv(x))
        x = self.head(x.flatten(start_dim=1))
        return x


# ----------- Backbone Class ----------- #
class RepConv(nn.Module):
    """A convolutional block that combines two convolution layers (kernel and point-wise).

    During training, uses parallel 3x3 + 1x1 branches for richer gradients.
    At inference, call :meth:`fuse` to merge both branches into a single 3x3
    convolution for +10-20% throughput with identical outputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        *,
        activation: Optional[str] = "SiLU",
        **kwargs,
    ):
        super().__init__()
        self.act = create_activation_function(activation)
        self.conv1 = Conv(in_channels, out_channels, kernel_size, activation=False, **kwargs)
        self.conv2 = Conv(in_channels, out_channels, 1, activation=False, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv1(x) + self.conv2(x))

    @staticmethod
    def _fuse_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
        """Fuse a Conv2d and BatchNorm2d into equivalent Conv2d weight and bias."""
        w = conv.weight
        mean, var, gamma, beta = bn.running_mean, bn.running_var, bn.weight, bn.bias
        std = (var + bn.eps).sqrt()
        fused_w = w * (gamma / std).reshape(-1, 1, 1, 1)
        fused_b = beta - mean * gamma / std
        return fused_w, fused_b

    def fuse(self) -> nn.Conv2d:
        """Fuse parallel 3x3 + 1x1 branches into a single 3x3 conv."""
        w3, b3 = self._fuse_bn(self.conv1.conv, self.conv1.bn)
        w1, b1 = self._fuse_bn(self.conv2.conv, self.conv2.bn)
        # Pad 1x1 kernel to 3x3
        w1_padded = F.pad(w1, [1, 1, 1, 1])
        # Create fused conv
        fused = nn.Conv2d(
            w3.shape[1], w3.shape[0], 3,
            stride=self.conv1.conv.stride, padding=self.conv1.conv.padding,
            groups=self.conv1.conv.groups, bias=True,
        )
        fused.weight.data = w3 + w1_padded
        fused.bias.data = b3 + b1
        return fused


class Bottleneck(nn.Module):
    """A bottleneck block with optional residual connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: Tuple[int, int] = (3, 3),
        residual: bool = True,
        expand: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        neck_channels = int(out_channels * expand)
        self.conv1 = RepConv(in_channels, neck_channels, kernel_size[0], **kwargs)
        self.conv2 = Conv(neck_channels, out_channels, kernel_size[1], **kwargs)
        self.residual = residual

        if residual and (in_channels != out_channels):
            self.residual = False
            logger.warning(
                "Residual connection disabled: in_channels ({}) != out_channels ({})", in_channels, out_channels
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.residual else y


class RepNCSP(nn.Module):
    """RepNCSP block with convolutions, split, and bottleneck processing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        csp_expand: float = 0.5,
        repeat_num: int = 1,
        neck_args: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()

        neck_channels = int(out_channels * csp_expand)
        self.conv1 = Conv(in_channels, neck_channels, kernel_size, **kwargs)
        self.conv2 = Conv(in_channels, neck_channels, kernel_size, **kwargs)
        self.conv3 = Conv(2 * neck_channels, out_channels, kernel_size, **kwargs)

        self.bottleneck = nn.Sequential(
            *[Bottleneck(neck_channels, neck_channels, **neck_args) for _ in range(repeat_num)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.bottleneck(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))


class ELAN(nn.Module):
    """ELAN  structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        *,
        process_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        if process_channels is None:
            process_channels = part_channels // 2

        self.conv1 = Conv(in_channels, part_channels, 1, **kwargs)
        self.conv2 = Conv(part_channels // 2, process_channels, 3, padding=1, **kwargs)
        self.conv3 = Conv(process_channels, process_channels, 3, padding=1, **kwargs)
        self.conv4 = Conv(part_channels + 2 * process_channels, out_channels, 1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(torch.cat([x1, x2, x3, x4], dim=1))
        return x5


class RepNCSPELAN(nn.Module):
    """RepNCSPELAN block combining RepNCSP blocks with ELAN structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        *,
        process_channels: Optional[int] = None,
        csp_args: Dict[str, Any] = {},
        csp_neck_args: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()

        if process_channels is None:
            process_channels = part_channels // 2

        self.conv1 = Conv(in_channels, part_channels, 1, **kwargs)
        self.conv2 = nn.Sequential(
            RepNCSP(part_channels // 2, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv(process_channels, process_channels, 3, padding=1, **kwargs),
        )
        self.conv3 = nn.Sequential(
            RepNCSP(process_channels, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv(process_channels, process_channels, 3, padding=1, **kwargs),
        )
        self.conv4 = Conv(part_channels + 2 * process_channels, out_channels, 1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(torch.cat([x1, x2, x3, x4], dim=1))
        return x5


class AConv(nn.Module):
    """Downsampling module combining average and max pooling with convolution for feature reduction."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_layer = {"kernel_size": 3, "stride": 2}
        self.avg_pool = Pool("avg", kernel_size=2, stride=1)
        self.conv = Conv(in_channels, out_channels, **mid_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        x = self.conv(x)
        return x


class ADown(nn.Module):
    """Downsampling module combining average and max pooling with convolution for feature reduction."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        half_in_channels = in_channels // 2
        half_out_channels = out_channels // 2
        mid_layer = {"kernel_size": 3, "stride": 2}
        self.avg_pool = Pool("avg", kernel_size=2, stride=1)
        self.conv1 = Conv(half_in_channels, half_out_channels, **mid_layer)
        self.max_pool = Pool("max", **mid_layer)
        self.conv2 = Conv(half_in_channels, half_out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv1(x1)
        x2 = self.max_pool(x2)
        x2 = self.conv2(x2)
        return torch.cat((x1, x2), dim=1)


class CBLinear(nn.Module):
    """Convolutional block that outputs multiple feature maps split along the channel dimension."""

    def __init__(self, in_channels: int, out_channels: List[int], kernel_size: int = 1, **kwargs):
        super(CBLinear, self).__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        self.conv = nn.Conv2d(in_channels, sum(out_channels), kernel_size, **kwargs)
        self.out_channels = list(out_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.conv(x)
        return x.split(self.out_channels, dim=1)


class SPPCSPConv(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels: int, out_channels: int, expand: float = 0.5, kernel_sizes: Tuple[int] = (5, 9, 13)):
        super().__init__()
        neck_channels = int(2 * out_channels * expand)
        self.pre_conv = nn.Sequential(
            Conv(in_channels, neck_channels, 1),
            Conv(neck_channels, neck_channels, 3),
            Conv(neck_channels, neck_channels, 1),
        )
        self.short_conv = Conv(in_channels, neck_channels, 1)
        self.pools = nn.ModuleList([Pool(kernel_size=kernel_size, stride=1) for kernel_size in kernel_sizes])
        self.post_conv = nn.Sequential(Conv(4 * neck_channels, neck_channels, 1), Conv(neck_channels, neck_channels, 3))
        self.merge_conv = Conv(2 * neck_channels, out_channels, 1)

    def forward(self, x):
        features = [self.pre_conv(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        features = torch.cat(features, dim=1)
        y1 = self.post_conv(features)
        y2 = self.short_conv(x)
        y = torch.cat((y1, y2), dim=1)
        return self.merge_conv(y)


class SPPELAN(nn.Module):
    """SPPELAN module comprising multiple pooling and convolution layers."""

    def __init__(self, in_channels: int, out_channels: int, neck_channels: Optional[int] = None):
        super(SPPELAN, self).__init__()
        neck_channels = neck_channels or out_channels // 2

        self.conv1 = Conv(in_channels, neck_channels, kernel_size=1)
        self.pools = nn.ModuleList([Pool("max", 5, stride=1) for _ in range(3)])
        self.conv5 = Conv(4 * neck_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        features = [self.conv1(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        return self.conv5(torch.cat(features, dim=1))


class UpSample(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.UpSample = nn.Upsample(**kwargs)

    def forward(self, x):
        return self.UpSample(x)


class CBFuse(nn.Module):
    def __init__(self, index: List[int], mode: str = "nearest"):
        super().__init__()
        self.idx = index
        self.mode = mode

    def forward(self, x_list: List[torch.Tensor]) -> List[Tensor]:
        target = x_list[-1]
        target_size = target.shape[2:]  # Batch, Channel, H, W

        res = [F.interpolate(x[pick_id], size=target_size, mode=self.mode) for pick_id, x in zip(self.idx, x_list)]
        out = torch.stack(res + [target]).sum(dim=0)
        return out


class ImplicitA(nn.Module):
    """
    Implement YOLOR - implicit knowledge(Add), paper: https://arxiv.org/abs/2105.04206
    """

    def __init__(self, channel: int, mean: float = 0.0, std: float = 0.02):
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std

        self.implicit = nn.Parameter(torch.empty(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.implicit + x


class ImplicitM(nn.Module):
    """
    Implement YOLOR - implicit knowledge(multiply), paper: https://arxiv.org/abs/2105.04206
    """

    def __init__(self, channel: int, mean: float = 1.0, std: float = 0.02):
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std

        self.implicit = nn.Parameter(torch.empty(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.implicit * x


# ----------- Efficient Modules (v9-swift) ----------- #
class GhostConv(nn.Module):
    """Ghost Convolution: generate more features from cheap depthwise linear operations.

    Produces half the features via a primary convolution, then generates the
    other half via a cheap depthwise 5x5 convolution. ~2x cheaper than
    standard Conv with minimal accuracy loss.

    Reference: Han et al., "GhostNet: More Features from Cheap Operations", CVPR 2020.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 1,
        stride: int = 1,
        activation: Optional[str] = "SiLU",
    ):
        super().__init__()
        hidden = out_channels // 2
        self.primary = Conv(in_channels, hidden, kernel_size, stride=stride, activation=activation)
        self.cheap = Conv(hidden, hidden, 5, stride=1, groups=hidden, activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        y = self.primary(x)
        return torch.cat([y, self.cheap(y)], dim=1)


class GhostELAN(nn.Module):
    """ELAN block using GhostConv for efficient feature aggregation.

    Maintains ELAN's multi-path gradient flow while being ~2x cheaper per
    processing branch thanks to GhostConv's depthwise feature generation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        *,
        process_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if process_channels is None:
            process_channels = part_channels // 2
        self.conv1 = Conv(in_channels, part_channels, 1, **kwargs)
        self.ghost1 = GhostConv(part_channels // 2, process_channels, 3)
        self.ghost2 = GhostConv(process_channels, process_channels, 3)
        self.conv_out = Conv(part_channels + 2 * process_channels, out_channels, 1, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.ghost1(x2)
        x4 = self.ghost2(x3)
        return self.conv_out(torch.cat([x1, x2, x3, x4], dim=1))


class SCDown(nn.Module):
    """Spatial-Channel decoupled downsampling.

    Decouples channel transformation (pointwise conv) from spatial
    downsampling (depthwise strided conv). More efficient than ADown's
    dual avg+max pooling paths while preserving more spatial information.

    Reference: YOLOv10 (Wang et al., Tsinghua, 2024).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t = 3, stride: int = 2):
        super().__init__()
        self.cv1 = Conv(in_channels, out_channels, 1, activation="SiLU")
        self.cv2 = Conv(out_channels, out_channels, kernel_size, stride=stride, groups=out_channels, activation=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.cv2(self.cv1(x))


class SE(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Recalibrates channel-wise feature responses by modelling channel
    interdependencies. Adds <1% FLOPs for a consistent +0.3-0.5% mAP gain.

    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, mid, 1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        w = self.sigmoid(self.fc2(self.act(self.fc1(self.pool(x)))))
        return x * w


class SpatialAttention(nn.Module):
    """Lightweight spatial attention from CBAM.

    Computes a spatial attention map from channel-wise max and average
    pooling, helping the detector focus on informative regions. Especially
    useful at the P3 scale for small-object detection.

    Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
    """

    def __init__(self, in_channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


# ----------- Transformer Modules (v9-vit) ----------- #
class TransformerBlock(nn.Module):
    """Standard pre-norm transformer encoder block.

    Consists of multi-head self-attention followed by an MLP (feed-forward
    network), both with residual connections and LayerNorm.
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerStage(nn.Module):
    """A stage of transformer blocks operating on 2D feature maps.

    Reshapes (B, C, H, W) -> (B, HW, C), applies a stack of transformer
    encoder blocks with global self-attention, then reshapes back. An
    optional 1x1 convolution projects channels when in != out.

    Best placed at stride >= 16 where the sequence length (HW) is small
    enough for efficient self-attention. At 640x384 input:
      - stride 16: 40x24 = 960 tokens  (efficient)
      - stride 32: 20x12 = 240 tokens  (very cheap)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 6,
        depth: int = 2,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.proj = Conv(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.blocks = nn.ModuleList([
            TransformerBlock(out_channels, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        return x
