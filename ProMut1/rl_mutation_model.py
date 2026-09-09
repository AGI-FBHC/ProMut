import argparse
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from kornia.filters import gaussian_blur2d
from atom_res_dict import abrev, label_res_dict, res_label_dict, letter1_3_dict
from model_promut1 import ProMutBackbone
from path_config import ANALYSIS_OUTPUT_DIR, FINETUNE_OUTPUT_DIR, PRETRAINED_MODEL, TRAIN_NPZ, VAL_NPZ


class LoRAConv3d(nn.Module):
    def __init__(self, conv: nn.Conv3d, rank=4, alpha=1.0):
        super().__init__()
        self.conv = conv
        self.rank = rank
        self.alpha = alpha

        # 完全匹配主干卷积的空间参数
        self.lora_A = nn.Conv3d(
            conv.in_channels,
            rank,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=False
        )

        self.lora_B = nn.Conv3d(
            rank,
            conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # nn.init.normal_(self.lora_A.weight, std=0.02)
        # nn.init.zeros_(self.lora_B.weight)

        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.conv(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.alpha / self.rank

        return out + lora_out


def add_lora_to_irmb(model, lora_rank=4, lora_alpha=1.0):
    for name, module in model.named_children():
        # 如果是Conv3d，直接替换为LoRAConv3d
        if isinstance(module, nn.Conv3d):
            setattr(model, name, LoRAConv3d(module, rank=lora_rank, alpha=lora_alpha))
        else:
            # 递归对子模块继续替换
            add_lora_to_irmb(module, lora_rank, lora_alpha)


class HybridMutationPredictor(nn.Module):
    def __init__(self, pretrained_path, num_classes=20, freeze_extractor=True, lora_rank=8, lora_alpha=32,hidden_dim = 256,dropout_rate=0.5):
        super(HybridMutationPredictor, self).__init__()

        # 加载预训练模型
        self.feature_extractor = self._load_pretrained_model(pretrained_path)

        # 判断特征维度
        feature_dim = 448 if self.feature_extractor.use_roi_pooling else 288

        # 冻结特征提取器参数
        if freeze_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("特征提取器参数已冻结")

        # 给主干加LoRA（假设add_lora_to_irmb函数已定义）
        add_lora_to_irmb(self.feature_extractor, lora_rank=lora_rank, lora_alpha=lora_alpha)


        # 简单线性分类头
        #self.classifier = nn.Linear(feature_dim, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            #nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def _load_pretrained_model(self, pretrained_path):

        model = ProMutBackbone(in_chans=7, num_classes=20, use_roi_pooling=True)
        if os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            print(f"成功加载预训练模型: {pretrained_path}")
        else:
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        # 移除分类头，只保留特征提取部分
        model.head = nn.Identity()
        return model

    def forward(self, x):
        features = self._extract_features(x)
        output = self.classifier(features)
        return output

    def _extract_features(self, x):
        # 如果特征提取器被冻结，使用torch.no_grad()避免计算梯度
        if not self.training and all(not p.requires_grad for p in self.feature_extractor.parameters()):
            with torch.no_grad():
                return self._forward_features(x)
        else:
            return self._forward_features(x)

    def _forward_features(self, x):
        x = self.feature_extractor.blk0(x)
        x = self.feature_extractor.down0(x)
        x = self.feature_extractor.blk1(x)
        if self.feature_extractor.use_roi_pooling:
            x = self.feature_extractor.down1(x)
            features_for_roi = x
            x = self.feature_extractor.blk2(x)
            x = self.feature_extractor.down2(x)
            x = self.feature_extractor.blk3(x)
            global_features = self.feature_extractor.global_max_pool(x)
            global_features = self.feature_extractor.flatten(global_features)
            voxel_features_list, voxel_coords_list, roi_centers_list = self.feature_extractor.extract_voxel_features(
                x, features_for_roi)
            if voxel_features_list and len(voxel_features_list) > 0:
                batch_roi_features = []
                for b in range(len(voxel_features_list)):
                    voxel_features = voxel_features_list[b]
                    voxel_coords = voxel_coords_list[b]
                    roi_centers = roi_centers_list[b]
                    if len(roi_centers) > 0:
                        roi_features = self.feature_extractor.voxel_roi_pooling(voxel_features, voxel_coords,
                                                                                roi_centers)
                        batch_roi_feature = torch.max(roi_features, dim=0)[0].unsqueeze(0)
                        batch_roi_features.append(batch_roi_feature)
                if batch_roi_features:
                    roi_features = torch.cat(batch_roi_features, dim=0)
                    features = torch.cat([global_features, roi_features], dim=1)
                else:
                    features = global_features
            else:
                features = global_features
        else:
            x = self.feature_extractor.down1(x)
            x = self.feature_extractor.blk2(x)
            x = self.feature_extractor.down2(x)
            x = self.feature_extractor.blk3(x)
            x = self.feature_extractor.global_max_pool(x)
            features = self.feature_extractor.flatten(x)
        return features


class SoftRank(nn.Module):
    def __init__(self, regularization_strength=1.0, epsilon=1e-10):
        super(SoftRank, self).__init__()
        self.regularization_strength = regularization_strength
        self.epsilon = epsilon

    def forward(self, scores):
        """
        计算软排名

        参数:
            scores: 形状为 [batch_size, num_classes] 的张量

        返回:
            soft_ranks: 形状为 [batch_size, num_classes] 的软排名张量
        """
        batch_size, num_classes = scores.shape

        # 计算所有可能的比较结果
        # 扩展维度以便进行广播
        s_i = scores.unsqueeze(2)  # [batch_size, num_classes, 1]
        s_j = scores.unsqueeze(1)  # [batch_size, 1, num_classes]

        # 计算 s_i > s_j 的软指示器
        # 使用 sigmoid 函数进行平滑化
        pairwise_comp = torch.sigmoid((s_i - s_j) / self.regularization_strength)

        # 计算每个元素的软排名（比它小的元素数量）
        # 减去0.5是为了处理自身比较的情况
        soft_ranks = pairwise_comp.sum(dim=2) - 0.5

        # 归一化排名到 [0, num_classes-1] 范围
        soft_ranks = soft_ranks / (num_classes - 1)

        return soft_ranks


class MutationDataset(Dataset):
    def __init__(self, npz_file, augment=False, augment_prob=0.5):
        # 加载数据
        data = np.load(npz_file, allow_pickle=True)
        self.data = data['data']
        self.positions = data['positions']
        self.protein_names = data['protein_names']
        self.mutation_labels = data['mutation_labels']
        self.mutation_types = data['mutation_types']
        self.ranked_types = data['ranked_types']
        self.augment = augment  # 是否使用数据增强
        self.augment_prob = augment_prob  # 数据增强概率

        # 确保数据格式正确
        if len(self.data.shape) == 5:  # 如果已经是[N, C, D, H, W]格式
            pass
        elif len(self.data.shape) == 6:  # 如果是[N, 1, C, D, H, W]格式
            self.data = self.data.squeeze(1)
        else:
            raise ValueError(f"Unexpected data shape: {self.data.shape}")

        # 保存原始数据的副本，用于每次重新应用增强
        if self.augment:
            self.original_data = self.data.copy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取输入数据 - 从原始数据中获取，而不是已经可能被增强过的数据
        if self.augment:
            x = torch.from_numpy(self.original_data[idx].astype(np.float32))

            # 每次访问时随机应用数据增强
            if np.random.rand() < self.augment_prob:
                # 增加掩盖方法的选择概率
                augment_probs = {'masking': 0.4, 'noise': 0.2, 'channel_specific': 0.2, 'flip': 0.2}
                augment_type = np.random.choice(
                    list(augment_probs.keys()),
                    p=list(augment_probs.values())
                )
                x = self._apply_single_augmentation(x, augment_type)
        else:
            x = torch.from_numpy(self.data[idx].astype(np.float32))

        # 获取突变标签和类型
        mutation_label = self.mutation_labels[idx]
        mutation_type = self.mutation_types[idx]
        ranked_type = self.ranked_types[idx]

        # 获取位置和蛋白质名称（用于记录）
        position = self.positions[idx]
        protein_name = self.protein_names[idx]

        # 直接返回原始数据，不再使用掩码
        return {
            'data': x,
            'mutation_label': mutation_label,  # 原始标签
            'mutation_type': mutation_type,  # 原始氨基酸类型
            'ranked_type': ranked_type,
            'position': position,
            'protein_name': protein_name
        }

    def _apply_single_augmentation(self, x, augment_type):
        """应用单一数据增强方法，避免过度变形"""
        # 原始数据形状: [7, 20, 20, 20]

        if augment_type == 'noise':
            # 添加轻微随机噪声 - 适用于所有通道
            noise_level = np.random.uniform(0.005, 0.02)  # 轻微噪声
            noise = torch.randn_like(x) * noise_level
            x = x + noise

        elif augment_type == 'channel_specific':
            # 对电荷和SASA通道进行特定增强 (通道5和6)
            # 对电荷通道应用小扰动
            charge_noise = torch.randn_like(x[5]) * 0.01
            x[5] = x[5] + charge_noise

            # 对SASA通道应用小扰动
            sasa_noise = torch.randn_like(x[6]) * 0.01
            x[6] = x[6] + sasa_noise

        elif augment_type == 'flip':
            # 随机翻转 - 只在一个维度上进行，保持结构的基本特性
            axis = np.random.randint(0, 3)  # 0, 1, 2分别对应x, y, z轴

            # 只对前5个通道（原子结构信息）应用翻转
            if axis == 0:
                x[:5] = x[:5].flip(1)  # 沿第一个空间维度翻转
            elif axis == 1:
                x[:5] = x[:5].flip(2)  # 沿第二个空间维度翻转
            elif axis == 2:
                x[:5] = x[:5].flip(3)  # 沿第三个空间维度翻转

        elif augment_type == 'masking':
            # 只在前5个通道的非零区域做mask
            structure = x[:5]  # [5, D, H, W]
            nonzero_coords = (structure != 0).nonzero(as_tuple=False)
            if nonzero_coords.size(0) > 0:
                # 随机选一个非零点
                idx = np.random.randint(0, nonzero_coords.size(0))
                c, x0, y0, z0 = nonzero_coords[idx].tolist()
                # 随机mask大小
                size = np.random.randint(3, 6)
                # 计算mask区域边界，防止越界
                x1 = max(0, x0 - size // 2)
                y1 = max(0, y0 - size // 2)
                z1 = max(0, z0 - size // 2)
                x2 = min(x.shape[1], x1 + size)
                y2 = min(x.shape[2], y1 + size)
                z2 = min(x.shape[3], z1 + size)
                # 对前5个通道mask
                x[:5, x1:x2, y1:y2, z1:z2] = 0
        elif augment_type == 'elastic_deformation':
            # 只对非零区域应用弹性变形
            # 获取非零区域的掩码
            nonzero_mask = (x[:5].abs().sum(dim=0) > 0).float()

            # 创建位移场 - 较小的位移以保持结构完整性
            displacement = torch.randn(3, *x.shape[1:]) * 0.5  # 控制变形程度

            # 应用高斯模糊使位移场平滑

            for i in range(3):
                displacement[i] = gaussian_blur2d(displacement[i].unsqueeze(0).unsqueeze(0),
                                                  kernel_size=(5, 5), sigma=(1.5, 1.5)).squeeze()

            # 创建网格
            d, h, w = x.shape[1:]
            z, y, x_grid = torch.meshgrid(torch.arange(d), torch.arange(h), torch.arange(w))

            # 应用位移
            z_new = (z + displacement[0] * nonzero_mask).clamp(0, d - 1)
            y_new = (y + displacement[1] * nonzero_mask).clamp(0, h - 1)
            x_new = (x_grid + displacement[2] * nonzero_mask).clamp(0, w - 1)

            # 使用网格采样进行变形
            # 注意：这需要实现一个3D网格采样函数，可以使用插值方法
            # 这里是简化版本，实际实现可能需要更复杂的插值
            x_deformed = x.clone()
            for c in range(5):  # 只对前5个通道应用
                for i in range(d):
                    for j in range(h):
                        for k in range(w):
                            if nonzero_mask[i, j, k] > 0:
                                # 获取新位置（四舍五入到最近的整数）
                                ni, nj, nk = int(z_new[i, j, k].item()), int(y_new[i, j, k].item()), int(
                                    x_new[i, j, k].item())
                                x_deformed[c, ni, nj, nk] = x[c, i, j, k]

            # 只替换前5个通道
            x[:5] = x_deformed[:5]

        elif augment_type == 'local_intensity':
            # 局部强度变化 - 只在非零区域应用
            nonzero_mask = (x[:5].abs().sum(dim=0) > 0)

            # 随机选择一个区域中心点
            nonzero_coords = nonzero_mask.nonzero(as_tuple=False)
            if nonzero_coords.size(0) > 0:
                idx = np.random.randint(0, nonzero_coords.size(0))
                i, j, k = nonzero_coords[idx].tolist()

                # 定义局部区域
                size = np.random.randint(3, 7)
                i1, j1, k1 = max(0, i - size // 2), max(0, j - size // 2), max(0, k - size // 2)
                i2, j2, k2 = min(x.shape[1], i1 + size), min(x.shape[2], j1 + size), min(x.shape[3], k1 + size)

                # 随机强度因子
                factor = np.random.uniform(0.8, 1.2)

                # 应用强度变化到前5个通道
                for c in range(5):
                    region = x[c, i1:i2, j1:j2, k1:k2]
                    # 只对非零值应用
                    nonzero_region = (region != 0)
                    if nonzero_region.sum() > 0:
                        x[c, i1:i2, j1:j2, k1:k2][nonzero_region] *= factor

        return x


class ProteinGroupedDataset(Dataset):
    """按蛋白质分组的数据集，每个样本是一个完整的蛋白质及其所有突变位点"""

    def __init__(self, npz_file, augment=False, augment_prob=0.5):
        # 加载数据
        data = np.load(npz_file, allow_pickle=True)
        self.data = data['data']
        self.positions = data['positions']
        self.protein_names = data['protein_names']
        self.mutation_labels = data['mutation_labels']
        self.mutation_types = data['mutation_types']
        self.ranked_types = data['ranked_types']
        self.augment = augment
        self.augment_prob = augment_prob

        # 确保数据格式正确
        if len(self.data.shape) == 5:  # 如果已经是[N, C, D, H, W]格式
            pass
        elif len(self.data.shape) == 6:  # 如果是[N, 1, C, D, H, W]格式
            self.data = self.data.squeeze(1)
        else:
            raise ValueError(f"Unexpected data shape: {self.data.shape}")

        # 按蛋白质名称分组
        self.protein_groups = {}
        for i, protein_name in enumerate(self.protein_names):
            if protein_name not in self.protein_groups:
                self.protein_groups[protein_name] = []
            self.protein_groups[protein_name].append(i)

        # 创建蛋白质列表
        self.unique_proteins = list(self.protein_groups.keys())
        print(f"数据集包含 {len(self.unique_proteins)} 个不同的蛋白质")

        # 保存原始数据的副本，用于每次重新应用增强
        if self.augment:
            self.original_data = self.data.copy()

    def __len__(self):
        return len(self.unique_proteins)

    def __getitem__(self, idx):
        # 获取蛋白质名称
        protein_name = self.unique_proteins[idx]

        # 获取该蛋白质的所有位点索引
        indices = self.protein_groups[protein_name]
        max_sites = 200  # 根据您的GPU内存调整
        if len(indices) > max_sites:
            indices = np.random.choice(indices, max_sites, replace=False)

        # 收集该蛋白质的所有数据
        protein_data = []
        protein_mutation_labels = []
        protein_mutation_types = []
        protein_ranked_types = []
        protein_positions = []

        for i in indices:
            # 获取输入数据
            if self.augment and np.random.rand() < self.augment_prob:
                x = torch.from_numpy(self.original_data[i].astype(np.float32))
                # 应用数据增强
                augment_types = ['masking', 'noise', 'channel_specific', 'flip']
                augment_type = np.random.choice(augment_types)
                x = self._apply_single_augmentation(x, augment_type)
            else:
                x = torch.from_numpy(self.data[i].astype(np.float32))

            protein_data.append(x)
            protein_mutation_labels.append(self.mutation_labels[i])
            protein_mutation_types.append(self.mutation_types[i])
            protein_ranked_types.append(self.ranked_types[i])
            protein_positions.append(self.positions[i])

        # 将数据堆叠成批次
        protein_data = torch.stack(protein_data)

        return {
            'data': protein_data,
            'mutation_label': protein_mutation_labels,
            'mutation_type': protein_mutation_types,
            'ranked_type': protein_ranked_types,
            'position': protein_positions,
            'protein_name': protein_name
        }

    def _apply_single_augmentation(self, x, augment_type):
        """应用单一数据增强方法，避免过度变形"""
        # 原始数据形状: [7, 20, 20, 20]

        if augment_type == 'noise':
            # 添加轻微随机噪声 - 适用于所有通道
            noise_level = np.random.uniform(0.005, 0.02)  # 轻微噪声
            noise = torch.randn_like(x) * noise_level
            x = x + noise

        elif augment_type == 'channel_specific':
            # 对电荷和SASA通道进行特定增强 (通道5和6)
            # 对电荷通道应用小扰动
            charge_noise = torch.randn_like(x[5]) * 0.01
            x[5] = x[5] + charge_noise

            # 对SASA通道应用小扰动
            sasa_noise = torch.randn_like(x[6]) * 0.01
            x[6] = x[6] + sasa_noise

        elif augment_type == 'flip':
            # 随机翻转 - 只在一个维度上进行，保持结构的基本特性
            axis = np.random.randint(0, 3)  # 0, 1, 2分别对应x, y, z轴

            # 只对前5个通道（原子结构信息）应用翻转
            if axis == 0:
                x[:5] = x[:5].flip(1)  # 沿第一个空间维度翻转
            elif axis == 1:
                x[:5] = x[:5].flip(2)  # 沿第二个空间维度翻转
            elif axis == 2:
                x[:5] = x[:5].flip(3)  # 沿第三个空间维度翻转

        elif augment_type == 'masking':
            # 随机掩盖部分体素
            # 选择一个随机区域进行掩盖
            mask_ratio = np.random.uniform(0.05, 0.15)  # 掩盖5%-15%的体素

            # 创建一个随机掩码
            mask_shape = x[0].shape  # [20, 20, 20]
            mask = torch.ones(mask_shape, device=x.device)

            # 随机选择一个起始点
            start_x = np.random.randint(0, mask_shape[0] - 5)
            start_y = np.random.randint(0, mask_shape[1] - 5)
            start_z = np.random.randint(0, mask_shape[2] - 5)

            # 随机选择掩盖区域的大小
            size_x = np.random.randint(3, 6)
            size_y = np.random.randint(3, 6)
            size_z = np.random.randint(3, 6)

            # 应用掩码
            mask[start_x:start_x + size_x, start_y:start_y + size_y, start_z:start_z + size_z] = 0

            # 对前5个通道应用掩码（原子结构信息）
            for i in range(5):
                x[i] = x[i] * mask

        return x


def protein_collate_fn(batch):
    """
    为按蛋白质分组的数据集设计的collate函数
    每个batch只包含一个蛋白质的所有位点
    """
    # 由于我们的batch_size=1（一个蛋白质一个batch），直接返回第一个元素
    return batch[0]


def custom_collate_fn(batch):
    # 提取可以直接堆叠的数据
    data = torch.stack([item['data'] for item in batch])

    # 对于不能直接堆叠的数据，保持列表形式
    mutation_labels = [item['mutation_label'] for item in batch]
    mutation_types = [item['mutation_type'] for item in batch]
    ranked_types = [item['ranked_type'] for item in batch]
    positions = [item['position'] for item in batch]
    protein_names = [item['protein_name'] for item in batch]

    return {
        'data': data,
        'mutation_label': mutation_labels,
        'mutation_type': mutation_types,
        'ranked_type': ranked_types,
        'position': positions,
        'protein_name': protein_names
    }


class CustomSpearmanLoss1(nn.Module):
    def __init__(self, regularization_strength=1.0):
        super(CustomSpearmanLoss1, self).__init__()
        self.soft_rank = SoftRank(regularization_strength)
        # 创建单字母到索引的映射
        self.letter_to_idx = {letter: res_label_dict[letter1_3_dict[letter]] for letter in abrev.values() if
                              letter in letter1_3_dict and letter1_3_dict[letter] in res_label_dict}

    def forward(self, predictions, mutation_labels, mutation_types, ranked_types=None):
        """
        计算预测和目标之间的软Spearman相关系数损失，优先使用ranked_type中的氨基酸排序
        """
        batch_size = predictions.size(0)
        device = predictions.device

        # 初始化损失
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_samples = 0

        # 对每个样本单独计算损失
        for i in range(batch_size):
            # 使用ranked_types（如果提供）
            if ranked_types is not None and len(ranked_types[i]) >= 2:
                # 从ranked_types中获取排序好的氨基酸列表
                ranked_aa_list = ranked_types[i]
                # print(ranked_aa_list)

                # 收集有效的预测值索引
                valid_indices = []
                for aa in ranked_aa_list:
                    if aa in self.letter_to_idx:
                        valid_indices.append(self.letter_to_idx[aa])
                # print(valid_indices)

                # 如果有效的氨基酸少于2个，无法计算相关系数，跳过此样本
                if len(valid_indices) < 2:
                    continue

                # 从predictions中获取对应索引的预测值
                pred = predictions[i, valid_indices]
                # print(predictions[i])
                # print(pred)

                # 计算预测值的软排名
                pred_ranks = self.soft_rank(pred.unsqueeze(0)).squeeze(0)
                # print(pred_ranks)

                # 创建目标排名（从0到n-1，表示从好到坏的排序）
                # ranked_types中的顺序已经是从好到坏排序的，所以目标排名就是从0开始的序列
                #target_ranks = torch.arange(len(valid_indices), device=device, dtype=torch.float32)
                target_ranks = (len(valid_indices) - 1) - torch.arange(len(valid_indices), device=device,
                                                                       dtype=torch.float32)
                # print(target_ranks)

                # 归一化目标排名到[0,1]范围，与软排名保持一致
                target_ranks = target_ranks / (len(valid_indices) - 1)

                # print(target_ranks)

                # 计算均值
                pred_mean = pred_ranks.mean()
                target_mean = target_ranks.mean()

                # 计算方差
                epsilon = 1e-8
                pred_var = ((pred_ranks - pred_mean) ** 2).mean() + epsilon
                target_var = ((target_ranks - target_mean) ** 2).mean() + epsilon

                # 计算协方差
                cov = ((pred_ranks - pred_mean) * (target_ranks - target_mean)).mean()

                # 计算Spearman相关系数
                correlation = cov / (torch.sqrt(pred_var * target_var))

                # 转换为损失（1 - 相关系数）
                correlation = torch.clamp(correlation, -1.0, 1.0)
                loss = - correlation

                total_loss = total_loss + loss
                valid_samples += 1

        # 如果没有有效样本，返回一个默认损失
        if valid_samples == 0:
            return torch.tensor(0.5, device=device, requires_grad=True)

        # 返回平均损失
        return total_loss / max(valid_samples, 1)


class CustomSpearmanLoss(nn.Module):
    def __init__(self, regularization_strength=1.0):
        super(CustomSpearmanLoss, self).__init__()
        self.soft_rank = SoftRank(regularization_strength)
        # 创建单字母到索引的映射
        self.letter_to_idx = {letter: res_label_dict[letter1_3_dict[letter]] for letter in abrev.values() if
                              letter in letter1_3_dict and letter1_3_dict[letter] in res_label_dict}

    def forward(self, predictions, mutation_labels, mutation_types, ranked_types=None):
        """
        计算预测和目标之间的软Spearman相关系数损失
        """
        num_positions = predictions.size(0)
        device = predictions.device
        
        # 初始化损失
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_positions = 0
        
        # 对每个位点单独计算损失
        for i in range(num_positions):
            # 使用ranked_types（如果提供）
            if ranked_types is not None and len(ranked_types[i]) >= 2:
                # 从ranked_types中获取排序好的氨基酸列表
                ranked_aa_list = ranked_types[i]
                
                # 收集有效的预测值索引
                valid_indices = []
                for aa in ranked_aa_list:
                    if aa in self.letter_to_idx:
                        valid_indices.append(self.letter_to_idx[aa])
                
                # 如果有效的氨基酸少于2个，无法计算相关系数，跳过此位点
                if len(valid_indices) < 2:
                    continue

                # 从predictions中获取对应索引的预测值
                pred = predictions[i, valid_indices]

                # 计算预测值的软排名
                #pred_ranks = self.soft_rank(pred.unsqueeze(0)).squeeze(0)

                sorted_indices = torch.argsort(pred, descending=True)
                # 创建一个空的排名张量
                pred_ranks = torch.empty_like(pred, dtype=torch.float32)
                # 根据排序后的索引填充排名
                # 最高分的排名是 len(pred) - 1，最低分的排名是 0
                rank_values = torch.arange(len(pred) - 1, -1, -1, device=pred.device, dtype=torch.float32)
                pred_ranks[sorted_indices] = rank_values

                target_ranks = (len(valid_indices) - 1) - torch.arange(len(valid_indices), device=device,
                                                                       dtype=torch.float32)

                # 归一化目标排名到[0,1]范围，与软排名保持一致
                #target_ranks = target_ranks / (len(valid_indices) - 1)

                # 计算均值
                pred_mean = pred_ranks.mean()
                target_mean = target_ranks.mean()

                # 计算方差
                epsilon = 1e-8
                pred_var = ((pred_ranks - pred_mean) ** 2).mean() + epsilon
                target_var = ((target_ranks - target_mean) ** 2).mean() + epsilon

                # 计算协方差
                cov = ((pred_ranks - pred_mean) * (target_ranks - target_mean)).mean()

                # 计算Spearman相关系数
                correlation = cov / (torch.sqrt(pred_var * target_var))

                # 转换为损失（1 - 相关系数）
                correlation = torch.clamp(correlation, -1.0, 1.0)
                loss = - correlation

                total_loss = total_loss + loss
                valid_positions += 1
        
        # 如果没有有效位点，返回一个默认损失
        if valid_positions == 0:
            return torch.tensor(0.5, device=device, requires_grad=True)

        # 返回该蛋白质所有位点的平均损失
        return total_loss / valid_positions


class BestLossPlateauScheduler:
    """基于最佳验证损失的学习率调度器"""

    def __init__(self, optimizer, factor=0.5, patience=5, verbose=False, min_lr=1e-8):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.all_without_improvement=0

    def step(self, val_loss):
        """根据验证损失更新学习率"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
            self.all_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            self.all_without_improvement += 1
        if self.all_without_improvement >= 2 * self.patience:
            if self.verbose:
                print(f"连续 {self.all_without_improvement} 个 epochs 损失没有改善，达到2倍patience，终止训练。")
            return True  # 返回 True 表示需要终止训练

        if self.epochs_without_improvement >= self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                if old_lr > self.min_lr:
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                    if self.verbose:
                        print(f"降低学习率: {old_lr:.2e} -> {new_lr:.2e}")

            self.epochs_without_improvement = 0
        return False


# Public ProMut name; retain the historical class for compatibility.
ProMutPredictor = HybridMutationPredictor


def train(args):
    seed = 12
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Loading datasets...")

    train_dataset = MutationDataset(args.train_data,augment=args.use_augmentation, augment_prob=args.augment_prob)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=custom_collate_fn)
    # train_dataset = ProteinGroupedDataset(args.train_data, augment=args.use_augmentation, augment_prob=args.augment_prob)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
    #                           num_workers=args.num_workers, collate_fn=protein_collate_fn)

    val_dataset = ProteinGroupedDataset(args.val_data)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, collate_fn=protein_collate_fn)
    print(f"Current train fold size: {len(train_dataset)}")
    print(f"验证数据集包含 {len(val_dataset)} 个蛋白质")
    print(f"Creating model from pretrained weights: {args.pretrained_model}")

    model = ProMutPredictor(
        pretrained_path=args.pretrained_model,
        freeze_extractor=args.freeze_extractor,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate
    ).to(device)
    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")
    if args.use_lora:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        print(f"Using LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}")
        # 计算LoRA参数数量
        lora_params = 0
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                lora_params += param.numel()

        print(f"LoRA parameters: {lora_params:,} ({lora_params / total_params:.2%} of total)")

        # 检查是否有非LoRA参数被错误地设置为可训练
        non_lora_trainable = trainable_params - lora_params
        if non_lora_trainable > 0:
            print(f"Warning: {non_lora_trainable:,} non-LoRA parameters are trainable!")
            # 打印这些参数的名称以便调试
            print("可训练的非LoRA参数:")
            for name, param in model.named_parameters():
                if param.requires_grad and 'lora_A' not in name and 'lora_B' not in name:
                    print(f"  - {name}: {param.numel():,} parameters")
    # 定义损失函数和优化器
    criterion1 = CustomSpearmanLoss1(regularization_strength=args.reg_strength)
    criterion2 = CustomSpearmanLoss(regularization_strength=args.reg_strength)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = BestLossPlateauScheduler(
        optimizer, factor=0.5, patience=5, verbose=True
    )

    # 训练记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    pseudo_epochs = args.pseudo_epochs
    pseudo_epochs =500
    plt.ion()
    plt.figure(figsize=(10, 6))

    # 训练循环
    print("Starting training...")
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        accumulation_steps = args.accumulation_steps  # 梯度累积步数
        optimizer.zero_grad()  # 在循环外清零梯度

        train_pbar = tqdm(train_loader, ncols=150,
                          desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for batch_idx, batch in enumerate(train_pbar):
            # 获取数据
            data = batch['data'].to(device)
            mutation_labels = batch['mutation_label']  # 列表
            mutation_types = batch['mutation_type']  # 列表
            ranked_types = batch['ranked_type']  # 列表

            outputs = model(data)
            loss = criterion1(outputs, mutation_labels, mutation_types, ranked_types)
            loss = loss / accumulation_steps
            loss.backward()
            display_loss = loss.item() * accumulation_steps
            train_loss += display_loss

            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()
                optimizer.zero_grad()

                correlation = - display_loss
                train_pbar.set_postfix({
                    'loss': f"{display_loss:.4f}",
                    'corr': f"{correlation:.4f}",
                    'update': f"{(batch_idx + 1) // accumulation_steps}"
                })
            else:
                train_pbar.set_postfix({
                    'loss': f"{display_loss:.4f}",
                    'accumulating': f"{(batch_idx + 1) % accumulation_steps}/{accumulation_steps}"
                })

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        model.eval()
        val_loss = 0.0
        val_protein_count = 0
        swa_val_loss = 0.0
        swa_active = args.use_swa and epoch >= args.swa_start_epoch
        # 保存蛋白质损失的目录
        protein_loss_dir = ANALYSIS_OUTPUT_DIR
        os.makedirs(protein_loss_dir, exist_ok=True)
        protein_loss_file = os.path.join(protein_loss_dir, "protein_losses.txt")
        # 数据分析目录
        data_analysis_dir = os.path.join(protein_loss_dir, "data_analysis")
        os.makedirs(data_analysis_dir, exist_ok=True)
        # 氨基酸预测结果字典
        protein_predictions = {}
        if epoch == 0:
            with open(protein_loss_file, 'w') as f:
                f.write("Protein_Name")
                for e in range(1, args.epochs + 1):
                    f.write(f"\tEpoch_{e}")
                f.write("\n")

        # 读取现有的蛋白质损失记录
        protein_losses_dict = {}
        if os.path.exists(protein_loss_file):
            with open(protein_loss_file, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:  # 跳过表头
                    parts = line.strip().split('\t')
                    if len(parts) > 0:
                        protein_name = parts[0]
                        losses = parts[1:]
                        # 确保列表长度足够存储所有epoch的损失
                        if len(losses) < args.epochs:
                            losses.extend([""] * (args.epochs - len(losses)))
                        protein_losses_dict[protein_name] = losses


        val_pbar = tqdm(val_loader, ncols=120, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                data = batch['data'].to(device)
                mutation_labels = batch['mutation_label']  # 列表
                mutation_types = batch['mutation_type']  # 列表
                ranked_types = batch['ranked_type']  # 列表
                protein_name = batch['protein_name']  # 字符串
                positions = batch['position']

                outputs = model(data)
                loss = criterion2(outputs, mutation_labels, mutation_types, ranked_types)

                if protein_name not in protein_losses_dict:
                    protein_losses_dict[protein_name] = [""] * args.epochs
                if epoch < args.epochs:
                    protein_losses_dict[protein_name][epoch] = f"{loss.item():.6f}"
                if protein_name not in protein_predictions:
                    protein_predictions[protein_name] = {}

                for i, position in enumerate(positions):
                    if position not in protein_predictions[protein_name]:
                        protein_predictions[protein_name][position] = {}
                    pos_probs = outputs[i].cpu().numpy()  # [20]
                    amino_acid_probs = {}
                    for aa_idx, prob in enumerate(pos_probs):
                        for aa_name, idx in res_label_dict.items():
                            if idx == aa_idx:
                                amino_acid_probs[aa_name] = float(prob)
                                break

                    protein_predictions[protein_name][position] = amino_acid_probs
                # 更新原始模型验证损失
                val_loss += loss.item()
                val_protein_count += 1
                correlation = - loss.item()
                # 更新进度条显示
                postfix_dict = {
                    'loss': f"{loss.item():.4f}",
                    'corr': f"{correlation:.4f}",
                    'protein': protein_name
                }
                val_pbar.set_postfix(postfix_dict)

        # 将蛋白质损失写回文件
        with open(protein_loss_file, 'w') as f:
            f.write("Protein_Name")
            for e in range(1, args.epochs + 1):
                f.write(f"\tEpoch_{e}")
            f.write("\n")

            for protein_name, losses in protein_losses_dict.items():
                f.write(protein_name)
                for i in range(args.epochs):
                    if i < len(losses) and losses[i]:
                        f.write(f"\t{losses[i]}")
                    else:
                        f.write("\t")
                f.write("\n")
        # 保存预测结果到文件
        prediction_file = os.path.join(data_analysis_dir, f"epoch_{epoch + 1}_predictions.txt")
        with open(prediction_file, 'w') as f:
            f.write("Protein_Name\tPosition\t")
            # 写入氨基酸表头
            aa_names = sorted(res_label_dict.keys())
            f.write("\t".join(aa_names))
            f.write("\n")

            for protein_name, positions_dict in protein_predictions.items():
                for position, aa_probs in positions_dict.items():
                    f.write(f"{protein_name}\t{position}\t")
                    # 按照表头顺序写入概率值
                    prob_values = [f"{aa_probs.get(aa_name, 0.0):.6f}" for aa_name in aa_names]
                    f.write("\t".join(prob_values))
                    f.write("\n")
        print(f"预测结果已保存到: {prediction_file}")
        avg_val_loss = val_loss / val_protein_count
        val_losses.append(avg_val_loss)
        # 更新学习率 - 先使用原始调度器
        if scheduler.step(avg_val_loss):
            print(f"Epoch {epoch+1}: 验证损失连续 {scheduler.patience * 2} 个epoch未改善，提前终止训练 。")
            break  # 终止训练循环
        # 打印训练和验证损失
        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        # 清除之前的图形并重新绘制
        plt.clf()
        plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        #plt.pause(0.01)
        plt.show()
        # 保存最佳原始模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"保存最佳原始模型，验证损失: {avg_val_loss:.4f}")
        # 每个epoch保存一次模型
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(args.output_dir, f'model_epoch_{epoch + 1}.pth'))
    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, os.path.join(args.output_dir, 'final_model.pth'))

    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train mutation predictor model')

    # 数据参数
    parser.add_argument('--augmented_folds_dir', type=str,
                        default='',
                        help='Directory containing augmented fold data')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of augmented folds to use')
    parser.add_argument(('--train_data'),type=str,
                        default=TRAIN_NPZ,
                        help='Path to the training data')
    parser.add_argument('--val_data', type=str,
                        default=VAL_NPZ,
                        help='Path to validation data')

    parser.add_argument('--pseudo_epochs', type=int, default=20)

    # 模型参数
    parser.add_argument('--pretrained_model', type=str,
                        default=PRETRAINED_MODEL,
                        help='Path to pretrained model')
    parser.add_argument('--freeze_extractor', action='store_true',
                        help='Freeze feature extractor parameters')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=640,
                        help='Hidden dimension of classifier')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='梯度累积步数，模拟的batch size = accumulation_steps * 实际batch_size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.00017887988460702127,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='Weight decay')
    parser.add_argument('--reg_strength', type=float, default=1.0,
                        help='Regularization strength for soft ranking')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA适配的秩')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA的alpha缩放因子')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout率，用于防止 过拟合')
    parser.add_argument('--augment_prob', type=float, default=0.5,
                        help='使用数据增强')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    # 添加LoRA相关参数
    parser.add_argument('--use_lora', action='store_true',
                        help='使用LoRA进行微调')
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=FINETUNE_OUTPUT_DIR,
                        help='Output directory for saving models')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save model every N epochs')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='使用数据增强')


    args = parser.parse_args()
    # 直接设置use_lora为True，无需命令行参数
    args.use_lora = True
    args.freeze_extractor = True
    args.use_swa = False
    args.use_augmentation = True

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
