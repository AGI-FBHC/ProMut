# ProMut: structure-pretrained local microenvironment modeling for protein mutation-effect prediction

<p align="center">
  <strong>English</strong> | <a href="README_cn.md">中文</a>
</p>

![ProMut framework](docs/framework.png)

<p align="center"><a href="docs/framework.pdf">Full-resolution framework PDF</a></p>

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
