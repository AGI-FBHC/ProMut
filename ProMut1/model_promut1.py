import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_, DropPath, make_divisible
from timm.layers.activations import Sigmoid, Swish, Mish, HardSigmoid, HardSwish, HardMish, Tanh, PReLU, GELU, sigmoid


def get_norm(norm_layer='in_1d'):
    eps = 1e-6
    norm_dict = {
        'none': nn.Identity,
        'in_1d': partial(nn.InstanceNorm1d, eps=eps),
        'in_2d': partial(nn.InstanceNorm2d, eps=eps),
        'in_3d': partial(nn.InstanceNorm3d, eps=eps),
        'bn_1d': partial(nn.BatchNorm1d, eps=eps),
        'bn_2d': partial(nn.BatchNorm2d, eps=eps),
        'bn_3d': partial(nn.BatchNorm3d, eps=eps),
        'gn': partial(nn.GroupNorm, eps=eps),
        'ln_1d': partial(nn.LayerNorm, eps=eps),
        'ln_2d': partial(LayerNorm, eps=eps),
        'ln_3d': partial(LayerNorm3d, eps=eps),
    }
    return norm_dict[norm_layer]


def get_act(act_layer='relu'):
    act_dict = {
        'none': nn.Identity,
        'sigmoid': Sigmoid,
        'swish': Swish,
        'mish': Mish,
        'hsigmoid': HardSigmoid,
        'hswish': HardSwish,
        'hmish': HardMish,
        'tanh': Tanh,
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'prelu': PReLU,
        'gelu': GELU,
        'silu': nn.SiLU
    }
    return act_dict[act_layer]


class ConvNormAct(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False,
                 skip=False, norm_layer='bn_3d', act_layer='relu', inplace=True, drop_path_rate=0.):
        super(ConvNormAct, self).__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv3d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = get_norm(norm_layer)(dim_out)
        self.act = get_act(act_layer)(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.conv_reduce = nn.Conv3d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv3d(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, x):
        x_se = x.mean((2, 3, 4), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)


class iRMB(nn.Module):

    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_3d',
                 act_layer='relu', v_proj=True, ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=64,
                 window_size=7, attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0.,
                 v_group=False, attn_pre=False, inplace=True):
        super().__init__()
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s

        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none',
                                  act_layer='none')
            self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias,
                                 norm_layer='none', act_layer=act_layer, inplace=inplace)
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none',
                                     act_layer=act_layer, inplace=inplace)
            else:
                self.v = nn.Identity()
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=ks, stride=stride, dilation=dilation,
                                      groups=1, norm_layer='bn_3d', act_layer='silu', inplace=inplace)
        self.se = SqueezeExcite(dim_mid, rd_ratio=se_ratio,
                                act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()#轻量化通道自注意力

        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)#投影层，无norm无激活函数
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W, D = x.shape
        if self.attn_s:
            # padding
            if self.window_size <= 0:
                window_size_W, window_size_H, window_size_D = W, H, D
            else:
                window_size_W, window_size_H, window_size_D = self.window_size, self.window_size, self.window_size
            pad_l, pad_t, pad_f = 0, 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            pad_be = (window_size_D - D % window_size_D) % window_size_D
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, pad_f, pad_be, 0, 0,))
            n1, n2, n3 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W, (D + pad_be) // window_size_D
            x = rearrange(x, 'b c (h1 n1) (w1 n2) (d1 n3) -> (b n1 n2 n3) c h1 w1 d1', n1=n1, n2=n2, n3=n3).contiguous()
            # attention
            b, c, h, w, d = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w d -> qk b heads (h w d) dim_head', qk=2, heads=self.num_head,
                           dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w d -> b heads (h w d) dim_head',
                              heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w d) dim_head -> b (heads dim_head) h w d', heads=self.num_head,
                                  h=h, w=w, d=d).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w d -> b heads (h w d) dim_head',
                              heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w d) dim_head -> b (heads dim_head) h w d', heads=self.num_head,
                                  h=h, w=w, d=d).contiguous()
            # unpadding
            x = rearrange(x_spa, '(b n1 n2 n3) c h1 w1 d1 -> b c (h1 n1) (w1 n2) (d1 n3)',
                          n1=n1, n2=n2, n3=n3).contiguous()
            if pad_r > 0 or pad_b > 0 or pad_be > 0:
                x = x[:, :, :H, :W, :D].contiguous()
        else:
            x = self.v(x)#如果没有自注意力机制就要通过v层

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

        x = self.proj_drop(x)
        x = self.proj(x)

        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, depth, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class LayerNorm3d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w d -> b h w d c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w d c -> b c h w d').contiguous()
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, stride=1, drop_path=0.):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_skip = in_chans == out_chans and stride == 1
        if not self.has_skip:
            self.skip = nn.Sequential(
                nn.Conv3d(in_chans, out_chans, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_chans)
            )

    def forward(self, x):
        if self.has_skip:
            return x + self.drop_path(self.conv(x))
        else:
            return self.skip(x) + self.drop_path(self.conv(x))


class VoxelQuery(nn.Module):
    """体素查询模块，用于查找邻近体素"""

    def __init__(self, radius=2, nsample=16):
        super(VoxelQuery, self).__init__()
        self.radius = radius
        self.nsample = nsample

    def forward(self, voxel_features, voxel_coords, roi_centers):

        # 计算曼哈顿距离
        M, _ = roi_centers.shape
        N, _ = voxel_coords.shape

        # 扩展维度以便进行广播
        roi_centers_expanded = roi_centers.unsqueeze(1)  # [M, 1, 3]
        voxel_coords_expanded = voxel_coords.unsqueeze(0)  # [1, N, 3]

        # 计算曼哈顿距离
        dist = torch.sum(torch.abs(roi_centers_expanded - voxel_coords_expanded), dim=-1)  # [M, N]

        # 找出每个ROI中心点附近的体素
        mask = dist < self.radius  # [M, N]

        # 对每个ROI，选择最近的nsample个体素
        _, idx = torch.topk(dist * mask.float() + (1e10) * (~mask).float(), k=self.nsample, dim=1,
                            largest=False)  # [M, nsample]

        # 收集选中体素的特征和坐标
        batch_indices = torch.arange(M, device=voxel_features.device).view(-1, 1).repeat(1,
                                                                                         self.nsample)  # [M, nsample]
        grouped_features = voxel_features[idx.view(-1)].view(M, self.nsample, -1)  # [M, nsample, C]
        grouped_coords = voxel_coords[idx.view(-1)].view(M, self.nsample, 3)  # [M, nsample, 3]

        return grouped_features, grouped_coords


class AcceleratedPointNet(nn.Module):
    """加速的PointNet模块，用于减少体素查询的计算复杂度"""

    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super(AcceleratedPointNet, self).__init__()
        if hidden_channels is None:
            hidden_channels = out_channels

        self.fc1 = nn.Conv1d(in_channels, hidden_channels, 1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.fc2 = nn.Conv1d(hidden_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = x.permute(0, 2, 1)  # [B, C, N]
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = torch.max(x, dim=2)[0]  # [B, C_out]
        return x


class VoxelRoIPooling(nn.Module):
    """体素RoI池化模块，用于从3D体素特征中提取RoI特征"""

    def __init__(self, feature_channels, out_channels, radius=2, nsample=16):
        super(VoxelRoIPooling, self).__init__()
        self.voxel_query = VoxelQuery(radius, nsample)
        self.pointnet = AcceleratedPointNet(feature_channels, out_channels)
        self.attention = nn.Sequential(
            nn.Conv1d(out_channels, out_channels // 4, 1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, features, coords, roi_centers):
        # 查询邻近体素
        grouped_features, _ = self.voxel_query(features, coords, roi_centers)

        # 应用PointNet处理
        roi_features = self.pointnet(grouped_features)

        return roi_features


class EMOCPD_VoxelRoI(nn.Module):
    """
    集成了Voxel RoI Pooling的EMOCPD模型
    """
    def __init__(self, in_chans=7, num_classes=20, drop_path=0.05, use_roi_pooling=True):
        super().__init__()

        dprs = [x.item() for x in torch.linspace(0, drop_path, 8)]
        self.use_roi_pooling = use_roi_pooling

        # 与原始EMOCPD相同的主干网络
        self.blk0 = nn.Sequential(
            iRMB(in_chans, 48, norm_in=True, has_skip=False, qkv_bias=True, attn_pre=True,
                 se_ratio=1, exp_ratio=1, dim_head=24, v_proj=False, attn_s=False, window_size=5),
            iRMB(48, 48, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[0],
                 se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5),
            iRMB(48, 48, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[1],
                 se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5),
        )
        self.down0 = iRMB(48, 72, norm_in=True, has_skip=False, qkv_bias=True, attn_pre=True, stride=2,
                          se_ratio=0, exp_ratio=2, dim_head=24, v_proj=True, attn_s=False, window_size=5)

        self.blk1 = nn.Sequential(
            iRMB(72, 72, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[2],
                 se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5),
            iRMB(72, 72, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[3],
                 se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5),
        )
        self.down1 = iRMB(72, 160, norm_in=True, has_skip=False, qkv_bias=True, attn_pre=True, stride=2,
                          se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5)

        # 添加Voxel RoI Pooling模块
        if self.use_roi_pooling:
            self.voxel_roi_pooling = VoxelRoIPooling(
                feature_channels=160,
                out_channels=160,
                radius=5,
                nsample=4
            )

            # 用于生成RoI中心点的注意力模块
            self.roi_attention = nn.Sequential(
                nn.Conv3d(160, 1, kernel_size=1),
                nn.Sigmoid()
            )

        self.blk2 = nn.Sequential(
            iRMB(160, 160, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[4],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=2, dim_head=32, v_proj=True, attn_s=True, window_size=5),
            iRMB(160, 160, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[5],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=2, dim_head=32, v_proj=True, attn_s=True, window_size=5),
        )
        self.down2 = iRMB(160, 288, norm_in=True, has_skip=False, qkv_bias=True, attn_pre=True, stride=2,
                          se_ratio=0, exp_ratio=2, dim_head=32, v_proj=True, attn_s=False, window_size=5)

        self.blk3 = nn.Sequential(
            iRMB(288, 288, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[6],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=1, dim_head=32, v_proj=True, attn_s=True, window_size=3),
            iRMB(288, 288, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[7],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=1, dim_head=32, v_proj=True, attn_s=True, window_size=3),
        )

        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        self.flatten = nn.Flatten()

        # 如果使用RoI Pooling，需要调整分类头的输入维度
        if self.use_roi_pooling:
            self.head = nn.Sequential(
                nn.Linear(288 + 160, 720),  # 增加了RoI特征的维度
                nn.ReLU(),
                nn.Linear(720, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(288, 720),
                nn.ReLU(),
                nn.Linear(720, num_classes)
            )

    def extract_voxel_features(self, x, features):
        """提取非空体素的特征和坐标"""
        batch_size, channels, depth, height, width = features.shape

        # 计算特征体的"能量"
        energy = torch.sum(torch.abs(features), dim=1)  # [B, D, H, W]

        # 找出能量大于阈值的体素（非空体素）
        #threshold = energy.mean() * 0.5
        threshold = 0.1
        masks = energy > threshold

        all_voxel_features = []
        all_voxel_coords = []
        all_roi_centers = []

        for b in range(batch_size):
            # 获取当前批次的掩码
            mask = masks[b]  # [D, H, W]

            # 找出非空体素的坐标
            indices = torch.nonzero(mask)  # [N, 3] - 每行是(d, h, w)坐标

            if len(indices) > 0:
                # 收集这些体素的特征
                d, h, w = indices[:, 0], indices[:, 1], indices[:, 2]
                feats = features[b, :, d, h, w].permute(1, 0)  # [N, C]

                # 使用注意力图来确定RoI中心
                attention_map = self.roi_attention(features[b:b + 1])  # [1, 1, D, H, W]

                # 找出注意力值最高的K个点作为RoI中心
                K = min(5, int(indices.size(0) * 0.1))  # 取10%的点或最多5个点
                attention_values = attention_map[0, 0, d, h, w]  # [N]
                _, top_indices = torch.topk(attention_values, k=K)

                roi_centers = indices[top_indices].float()  # [K, 3]

                all_voxel_features.append(feats)
                all_voxel_coords.append(indices.float())
                all_roi_centers.append(roi_centers)

        return all_voxel_features, all_voxel_coords, all_roi_centers

    def forward(self, x):
        # 前向传播到blk1
        x = self.blk0(x)
        x = self.down0(x)
        x = self.blk1(x)

        # 应用Voxel RoI Pooling
        if self.use_roi_pooling:
            # 继续前向传播
            x = self.down1(x)
            # 保存中间特征用于后续处理
            features_for_roi = x

            x = self.blk2(x)
            x = self.down2(x)
            x = self.blk3(x)

            # 提取全局特征
            global_features = self.global_max_pool(x)
            global_features = self.flatten(global_features)

            # 提取体素特征和坐标
            voxel_features_list, voxel_coords_list, roi_centers_list = self.extract_voxel_features(x, features_for_roi)

            # 如果有有效的RoI
            if voxel_features_list:
                batch_roi_features = []

                # 对每个批次处理RoI
                for b in range(len(voxel_features_list)):
                    voxel_features = voxel_features_list[b]
                    voxel_coords = voxel_coords_list[b]
                    roi_centers = roi_centers_list[b]

                    # 应用Voxel RoI Pooling
                    roi_features = self.voxel_roi_pooling(voxel_features, voxel_coords, roi_centers)

                    # 对所有RoI特征进行最大池化，得到每个批次的RoI特征
                    batch_roi_feature = torch.max(roi_features, dim=0)[0].unsqueeze(0)  # [1, C]
                    batch_roi_features.append(batch_roi_feature)

                # 拼接所有批次的RoI特征
                roi_features = torch.cat(batch_roi_features, dim=0)  # [B, C]

                # 拼接全局特征和RoI特征
                combined_features = torch.cat([global_features, roi_features], dim=1)

                # 分类
                x = self.head(combined_features)
            else:
                # 如果没有有效的RoI，只使用全局特征
                x = self.head(global_features)
        else:
            # 原始EMOCPD的前向传播
            x = self.down1(x)
            x = self.blk2(x)
            x = self.down2(x)
            x = self.blk3(x)
            x = self.global_max_pool(x)
            x = self.flatten(x)
            x = self.head(x)

        return x


class EMOCPD_NoIRMB(nn.Module):
    def __init__(self, in_chans=7, num_classes=20, drop_path=0.05):
        super().__init__()

        dprs = [x.item() for x in torch.linspace(0, drop_path, 8)]

        # 替换iRMB为普通卷积块
        self.blk0 = nn.Sequential(
            ConvBlock(in_chans, 48, stride=1),
            ConvBlock(48, 48, drop_path=dprs[0]),
            ConvBlock(48, 48, drop_path=dprs[1])
        )
        self.down0 = ConvBlock(48, 72, stride=2)

        self.blk1 = nn.Sequential(
            ConvBlock(72, 72, drop_path=dprs[2]),
            ConvBlock(72, 72, drop_path=dprs[3])
        )
        self.down1 = ConvBlock(72, 160, stride=2)

        # 保留MHSA iRMB
        self.blk2 = nn.Sequential(
            iRMB(160, 160, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[4],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=2, dim_head=32, v_proj=True, attn_s=True, window_size=5),
            iRMB(160, 160, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[5],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=2, dim_head=32, v_proj=True, attn_s=True, window_size=5)
        )
        self.down2 = ConvBlock(160, 288, stride=2)

        self.blk3 = nn.Sequential(
            iRMB(288, 288, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[6],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=1, dim_head=32, v_proj=True, attn_s=True, window_size=3),
            iRMB(288, 288, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[7],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=1, dim_head=32, v_proj=True, attn_s=True, window_size=3)
        )

        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(288, 720),
            nn.ReLU(),
            nn.Linear(720, num_classes)
        )

    def forward(self, x):
        x = self.blk0(x)
        x = self.down0(x)

        x = self.blk1(x)
        x = self.down1(x)

        x = self.blk2(x)
        x = self.down2(x)

        x = self.blk3(x)
        x = self.global_max_pool(x)
        x = self.flatten(x)

        x = self.head(x)

        return x


class EMOCPD_NoMHSA(nn.Module):
    def __init__(self, in_chans=7, num_classes=20, drop_path=0.05):
        super().__init__()

        dprs = [x.item() for x in torch.linspace(0, drop_path, 8)]

        # 保留普通iRMB
        self.blk0 = nn.Sequential(
            iRMB(in_chans, 48, norm_in=True, has_skip=False, qkv_bias=True, attn_pre=True,
                 se_ratio=1, exp_ratio=1, dim_head=24, v_proj=False, attn_s=False, window_size=5),
            iRMB(48, 48, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[0],
                 se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5),
            iRMB(48, 48, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[1],
                 se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5)
        )
        self.down0 = iRMB(48, 72, norm_in=True, has_skip=False, qkv_bias=True, attn_pre=True, stride=2,
                          se_ratio=0, exp_ratio=2, dim_head=24, v_proj=True, attn_s=False, window_size=5)

        self.blk1 = nn.Sequential(
            iRMB(72, 72, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[2],
                 se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5),
            iRMB(72, 72, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[3],
                 se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5)
        )
        self.down1 = iRMB(72, 160, norm_in=True, has_skip=False, qkv_bias=True, attn_pre=True, stride=2,
                          se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5)

        # 将MHSA iRMB替换为普通iRMB (关键是将attn_s设为False)
        self.blk2 = nn.Sequential(
            iRMB(160, 160, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[4],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=2, dim_head=32, v_proj=True, attn_s=False, window_size=5),
            iRMB(160, 160, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[5],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=2, dim_head=32, v_proj=True, attn_s=False, window_size=5)
        )
        self.down2 = iRMB(160, 288, norm_in=True, has_skip=False, qkv_bias=True, attn_pre=True, stride=2,
                          se_ratio=0, exp_ratio=2, dim_head=32, v_proj=True, attn_s=False, window_size=5)

        self.blk3 = nn.Sequential(
            iRMB(288, 288, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[6],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=1, dim_head=32, v_proj=True, attn_s=False, window_size=3),
            iRMB(288, 288, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[7],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=1, dim_head=32, v_proj=True, attn_s=False, window_size=3)
        )

        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(288, 720),
            nn.ReLU(),
            nn.Linear(720, num_classes)
        )

    def forward(self, x):
        x = self.blk0(x)
        x = self.down0(x)

        x = self.blk1(x)
        x = self.down1(x)

        x = self.blk2(x)
        x = self.down2(x)

        x = self.blk3(x)
        x = self.global_max_pool(x)
        x = self.flatten(x)

        x = self.head(x)

        return x


class EMOCPD_NoIRMB_NoMHSA(nn.Module):
    def __init__(self, in_chans=7, num_classes=20, drop_path=0.05):
        super().__init__()

        dprs = [x.item() for x in torch.linspace(0, drop_path, 8)]

        # 全部替换为普通卷积块
        self.blk0 = nn.Sequential(
            ConvBlock(in_chans, 48, stride=1),
            ConvBlock(48, 48, drop_path=dprs[0]),
            ConvBlock(48, 48, drop_path=dprs[1])
        )
        self.down0 = ConvBlock(48, 72, stride=2)

        self.blk1 = nn.Sequential(
            ConvBlock(72, 72, drop_path=dprs[2]),
            ConvBlock(72, 72, drop_path=dprs[3])
        )
        self.down1 = ConvBlock(72, 160, stride=2)

        self.blk2 = nn.Sequential(
            ConvBlock(160, 160, drop_path=dprs[4]),
            ConvBlock(160, 160, drop_path=dprs[5])
        )
        self.down2 = ConvBlock(160, 288, stride=2)

        self.blk3 = nn.Sequential(
            ConvBlock(288, 288, drop_path=dprs[6]),
            ConvBlock(288, 288, drop_path=dprs[7])
        )

        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(288, 720),
            nn.ReLU(),
            nn.Linear(720, num_classes)
        )

    def forward(self, x):
        x = self.blk0(x)
        x = self.down0(x)

        x = self.blk1(x)
        x = self.down1(x)

        x = self.blk2(x)
        x = self.down2(x)

        x = self.blk3(x)
        x = self.global_max_pool(x)
        x = self.flatten(x)

        x = self.head(x)

        return x


# Public ProMut name; retain the historical class for compatibility.
ProMutBackbone = EMOCPD_VoxelRoI
