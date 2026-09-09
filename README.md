<a id="top"></a>

# ProMut: structure-pretrained local microenvironment modeling for protein mutation-effect prediction

<p align="center">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</p>

![ProMut framework](docs/framework.png)

<p align="center"><a href="docs/framework.pdf">Full-resolution PDF / 高清框架 PDF</a></p>

<a id="english"></a>

## English

ProMut predicts protein mutation effects through residue-centered microenvironment construction, structural pretraining on natural proteins, and DMS-supervised LoRA fine-tuning.

### Installation

```bash
git clone https://github.com/AGI-FBHC/ProMut.git
cd ProMut
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

PDB preprocessing additionally requires `requirements-preprocess.txt`. GPU training is recommended.

### Data and weights

Datasets and model weights are available by request. Email [luochenxi@stu.jiangnan.cn](mailto:luochenxi@stu.jiangnan.cn) and specify the required files.

| Resource | Purpose |
|---|---|
| `data_csv/` | Natural protein structure CSVs for pretraining |
| `promut2_dataset/`, `promut2_csv/` | PDBs and processed structures associated with DMS proteins |
| `train_data/`, `data_test_0.npz`, `label_test_0.npz` | Pretraining and structural evaluation data |
| `train_0613.npz`, `val_0613.npz` | Prepared DMS fine-tuning data |
| `best_7093model.pth.tar` | Pretrained backbone checkpoint |

Place the loadable checkpoint at `ProMut1/best_7093model.pth.tar`. An extracted checkpoint folder cannot be passed to `torch.load`.

### Training workflow

#### 1. Local microenvironment construction

PDB2PQR and FreeSASA provide atomic charge and SASA. Each residue is represented as a `7 × 20 × 20 × 20` voxel grid.

```bash
python ProMut1/data_utils.py
python ProMut1/generate_full_sidechain_box_20A.py
python ProMut1/build_pretrain_shards.py
```

The scripts generate structure CSVs, residue voxel files, and 20 pairs of `data_*.npz` / `label_*.npz` shards. Paths are defined in `ProMut1/path_config.py`.

#### 2. Structural pretraining

```bash
python ProMut1/train.py
```

The `ProMutBackbone` predicts the central wild-type amino acid as a 20-class task.

| Setting | Value |
|---|---:|
| Loss | Cross-entropy |
| Epochs / batch size | `8 / 150` |
| Optimizer | Adam |
| Learning rate / weight decay | `1e-5 / 1e-3` |

Checkpoints are saved in `model_save/EMOCPD_VoxelRoI/`.

#### 3. DMS fine-tuning

```bash
python ProMut1/rl_mutation_model.py --train_data train_0613.npz --val_data val_0613.npz --pretrained_model ProMut1/best_7093model.pth.tar --output_dir model_save/ProMut_2
```

The pretrained backbone is frozen and adapted with LoRA to align predicted mutation rankings with experimental DMS rankings.

| Setting | Value |
|---|---:|
| Epochs / batch size | `500 / 200` |
| Optimizer | AdamW |
| Learning rate / weight decay | `1.7887988460702127e-4 / 3e-4` |
| LoRA rank / alpha | `16 / 32` |
| Dropout / augmentation probability | `0.5 / 0.5` |

Outputs include `best_model.pth`, periodic checkpoints, `final_model.pth`, and per-protein records under `analysis_outputs/`.

### Experiments

**TS50 / TS500 sequence recovery.** Predict the wild-type residue from its local structure and report Top-1 recovery accuracy.

```bash
python ProMut1/predict_emocpd.py --protein-csv testset_TS50_csv/csv/protein.csv --model ProMut1/best_7093model.pth.tar --chain A --topk 20 --out predictions/TS50_protein.csv
```

**DMS mutation ranking.** Fine-tuning evaluates `val_0613.npz` protein by protein using the rank-correlation loss implemented in `rl_mutation_model.py`.

`predict_emocpd.py` loads the pretrained backbone and is not a standalone inference entry point for the fine-tuned `best_model.pth`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="中文"></a>

## 中文

ProMut 包含三个阶段：以残基为中心构建局部微环境、在天然蛋白质上进行结构预训练，以及利用 DMS 实验数据进行 LoRA 微调。

### 环境安装

```bash
git clone https://github.com/AGI-FBHC/ProMut.git
cd ProMut
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

从 PDB 开始预处理还需安装 `requirements-preprocess.txt`。完整训练建议使用 GPU。

### 数据与模型

数据集和模型权重可通过邮件申请。请联系 [luochenxi@stu.jiangnan.cn](mailto:luochenxi@stu.jiangnan.cn)，并注明所需文件。

| 资源 | 用途 |
|---|---|
| `data_csv/` | 天然蛋白质结构 CSV，用于预训练 |
| `promut2_dataset/`、`promut2_csv/` | DMS 蛋白对应的 PDB 和处理后结构 |
| `train_data/`、`data_test_0.npz`、`label_test_0.npz` | 预训练和结构评估数据 |
| `train_0613.npz`、`val_0613.npz` | 已处理的 DMS 微调数据 |
| `best_7093model.pth.tar` | 预训练骨干权重 |

将可加载的权重放入 `ProMut1/best_7093model.pth.tar`。解压后的权重文件夹不能直接传给 `torch.load`。

### 训练流程

#### 1. 局部微环境构建

使用 PDB2PQR 和 FreeSASA 获得原子电荷与 SASA，并将每个残基表示为 `7 × 20 × 20 × 20` 体素。

```bash
python ProMut1/data_utils.py
python ProMut1/generate_full_sidechain_box_20A.py
python ProMut1/build_pretrain_shards.py
```

依次生成结构 CSV、残基体素和 20 对 `data_*.npz` / `label_*.npz` 分片，路径由 `ProMut1/path_config.py` 配置。

#### 2. 天然蛋白质结构预训练

```bash
python ProMut1/train.py
```

`ProMutBackbone` 根据局部结构预测中心残基所属的 20 类野生型氨基酸。

| 设置 | 数值 |
|---|---:|
| 损失 | 交叉熵 |
| Epoch / Batch size | `8 / 150` |
| 优化器 | Adam |
| 学习率 / 权重衰减 | `1e-5 / 1e-3` |

权重保存至 `model_save/EMOCPD_VoxelRoI/`。

#### 3. DMS 突变数据微调

```bash
python ProMut1/rl_mutation_model.py --train_data train_0613.npz --val_data val_0613.npz --pretrained_model ProMut1/best_7093model.pth.tar --output_dir model_save/ProMut_2
```

冻结预训练骨干并加入 LoRA，使预测突变排序接近实验 DMS 排序。

| 设置 | 数值 |
|---|---:|
| Epoch / Batch size | `500 / 200` |
| 优化器 | AdamW |
| 学习率 / 权重衰减 | `1.7887988460702127e-4 / 3e-4` |
| LoRA rank / alpha | `16 / 32` |
| Dropout / 数据增强概率 | `0.5 / 0.5` |

输出包括 `best_model.pth`、定期权重、`final_model.pth`，以及 `analysis_outputs/` 中的逐蛋白记录。

### 实验设置

**TS50 / TS500 序列恢复实验。** 根据局部结构预测野生型残基，以 Top-1 序列恢复准确率评价模型。

```bash
python ProMut1/predict_emocpd.py --protein-csv testset_TS50_csv/csv/protein.csv --model ProMut1/best_7093model.pth.tar --chain A --topk 20 --out predictions/TS50_protein.csv
```

**DMS 突变效应排序实验。** 微调过程中按蛋白质验证 `val_0613.npz`，使用 `rl_mutation_model.py` 中实现的排序相关损失。

`predict_emocpd.py` 加载的是预训练骨干，不能作为 DMS 微调输出 `best_model.pth` 的独立推理入口。

<p align="right"><a href="#top">↑ 返回顶部</a></p>
