# ProMut

ProMut predicts protein mutation effects from local three-dimensional structural environments. This source release follows the supplied ProMut manuscript and the original project README: structural pretraining with iRMB and VoxelRoI, followed by DMS-supervised LoRA fine-tuning with a differentiable rank-based loss.

这是根据论文和原项目 README 整理的代码发布目录。包含预处理、预训练、预训练模型推理、LoRA 微调及现有消融网络；不包含论文文稿、大型数据集、模型权重和历史实验输出。论文对应关系、删减依据和复现限制见 [整理说明](docs/整理说明.md)。本版本未重新训练或复现论文指标。

## 目录

```text
ProMut/
├── README.md
├── requirements.txt
├── requirements-preprocess.txt
├── .gitignore
├── ProMut1/
│   ├── atom_res_dict.py                     # 原子、残基及标签字典
│   ├── data_utils.py                        # PDB → CSV、单残基体素输入
│   ├── generate_full_sidechain_box_20A.py    # CSV → 7×20×20×20 体素
│   ├── build_pretrain_shards.py             # 合并预训练 NPZ
│   ├── datasets.py                         # 预训练数据加载器
│   ├── model_promut1.py                     # 主模型及已有消融模型
│   ├── train.py                            # 预训练入口
│   ├── predict_emocpd.py                    # 预训练模型推理入口
│   ├── rl_mutation_model.py                 # LoRA 微调入口
│   └── path_config.py                       # 项目相对路径
├── tests/smoke.py
└── docs/                                   # 整理依据、校验记录、数据要求
```

对外统一使用以下 ProMut 模型名称，历史名称继续兼容：

| 阶段 | 对外名称 | 兼容名称 |
|---|---|---|
| 结构预训练与结构预测 | `ProMutBackbone` | `EMOCPD_VoxelRoI` |
| DMS LoRA 微调 | `ProMutPredictor` | `HybridMutationPredictor` |

新旧名称引用同一个类，不添加包装层，不改变模型层名、参数或 `state_dict` 键。训练和预测入口使用对外名称；文件名、命令行参数和历史输出目录保持兼容。

在 `ProMut1` 模块可导入的环境中使用：

```python
from model_promut1 import ProMutBackbone
from rl_mutation_model import ProMutPredictor

backbone = ProMutBackbone(in_chans=7, num_classes=20, use_roi_pooling=True)
predictor = ProMutPredictor(pretrained_path="ProMut1/best_7093model.pth.tar")
```

其中微调模型需要实际存在的预训练权重；旧名称的导入和调用仍然可用。

## 环境安装

已在现有 Python 3.8.20、PyTorch 2.4.1 环境中进行 CPU 冒烟验证。以下是在同版本环境中安装 CPU 依赖的命令；GPU 用户应安装适合自己 CUDA 环境的同版本 PyTorch。

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

从原始 PDB 开始预处理还需要：

```bash
python -m pip install -r requirements-preprocess.txt
```

`freesasa` 和 `pdb2pqr` 仅用于 PDB 转 CSV。FreeSASA 在 Windows 上可能需要本地编译工具，原项目也记录了中文路径兼容问题；遇到此问题可将运行副本放在英文路径。`PDB2PQR_BIN` 可指定 PDB2PQR 可执行文件。当前测试使用已有环境，未验证从空环境安装和完整 PDB 预处理。

## 准备数据与权重

所有命令从本目录执行。将需要的数据放入以下位置（目录按需创建），或修改 `ProMut1/path_config.py`：

| 文件或目录 | 用途 |
|---|---|
| `data_csv/*.csv` | 预训练结构 CSV |
| `promut2_dataset/*.pdb` | 待处理 PDB |
| `promut2_csv/csv/*.csv` | PDB 转换后的结构 CSV |
| `pretrain_npz/*.npz` | 按残基类型保存的体素 |
| `pretrain_shards/data_0.npz` 至 `data_19.npz` | 预训练输入 |
| `pretrain_shards/label_0.npz` 至 `label_19.npz` | 预训练标签 |
| `data_test_0.npz`、`label_test_0.npz` | 可选预训练评估集 |
| `ProMut1/best_7093model.pth.tar` | 原项目预训练权重 |
| `train_0613.npz`、`val_0613.npz` | 已处理的微调训练与验证数据 |

发布目录不复制这些大型文件。数据结构见 [数据说明](docs/DATA.md)。目前没有提供公开下载地址；用户需自行准备原项目数据和权重。微调数据的原始生成脚本未在提供的项目代码中找到，因此不能仅凭本目录从原始 DMS 标签完整重建微调集。

## 运行流程

### PDB 转结构 CSV

```bash
python ProMut1/data_utils.py
```

读取 `promut2_dataset/`，写入 `promut2_csv/`。这一步与预训练体素生成的输入目录不同；如需对这些 CSV 生成预训练体素，请将 `path_config.py` 中的 `DATA_CSV_DIR` 指向 `promut2_csv/csv/`。不要把待评估蛋白混入预训练数据。

### 预训练

```bash
python ProMut1/generate_full_sidechain_box_20A.py
python ProMut1/build_pretrain_shards.py
python ProMut1/train.py
```

前两步依次从 `data_csv/` 写入 `pretrain_npz/` 和 `pretrain_shards/`；训练入口已统一读取 `pretrain_shards/`。合并脚本按氨基酸类型生成 20 个文件，缺少某类型时会跳过；训练前应确认 20 对文件齐全。它会一次加载某类型的所有体素，数据量较大时需要充足内存。

预训练参数保留在 `train.py` 中（batch size 150、8 epochs 等），输出为 `model_save/EMOCPD_VoxelRoI/`。自动推理默认权重文件名不会随训练结果更新，请通过 `--model` 指定新权重。缺少测试数据时不报告测试准确率，仍保存每轮权重。

### 预训练模型推理

```bash
python ProMut1/predict_emocpd.py --help
python ProMut1/predict_emocpd.py --protein-csv promut2_csv/csv/your_protein.csv --model ProMut1/best_7093model.pth.tar --chain A --topk 20 --out predictions/your_protein_predictions.csv
```

输出各残基的氨基酸预测分数/排名。可用 `--effect-out` 导出基于预训练输出计算的突变效应表。该入口加载 `state_dict` 格式的预训练模型，**不能直接加载微调后的 `best_model.pth`**，也不能将其输出视为论文微调结果。

### DMS LoRA 微调

```bash
python ProMut1/rl_mutation_model.py --help
python ProMut1/rl_mutation_model.py --train_data train_0613.npz --val_data val_0613.npz --pretrained_model ProMut1/best_7093model.pth.tar --output_dir model_save/ProMut_2
```

默认保持原训练行为：冻结特征提取器参数、添加 LoRA、启用数据增强。支持 `--epochs`、`--batch_size`、`--learning_rate`、`--lora_rank`、`--lora_alpha` 等。训练过程中按蛋白质验证，输出权重到指定目录，分析记录在 `analysis_outputs/`。缺少预训练权重时明确报错，避免误用随机初始化进行微调。

当前脚本中的 `--use_lora`、`--freeze_extractor` 和 `--use_augmentation` 为历史兼容参数，主入口始终开启这些设置，不能用于关闭相应功能。完整消融训练入口和独立微调推理入口未包含在原 README 主流程中，不能直接由已有模型名称推断论文消融实验已可一键复现。

## 验证

```bash
python tests/smoke.py
python tests/smoke.py --checkpoint ProMut1/best_7093model.pth.tar
```

第一条检查主模型/现有消融模型输出和语法；第二条额外检查真实预训练权重加载及 LoRA 梯度。整理时还验证了新旧四个模型在相同输入与权重下的输出完全一致，记录见 `docs/validation.json`。这些检查不替代完整训练、数据集划分核验或论文指标复现。

## GitHub 发布

上传本目录下的源码、README、依赖、测试和 docs 即可；`.gitignore` 已排除权重、数据、缓存及运行输出。本次整理未创建远程仓库或执行上传。

作者、论文正式题名/DOI、数据下载地址和第三方来源授权尚需由维护者补充。未擅自指定 MIT 等开源许可证；公开发布前应确认项目代码及继承实现的授权，再添加适用的 `LICENSE` 和引用信息。
