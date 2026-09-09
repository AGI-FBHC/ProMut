# ProMut: structure-pretrained local microenvironment modeling for protein mutation-effect prediction

<p align="center">
  <a href="README.md">English</a> | <a href="README_cn.md">中文</a>
</p>

Protein mutation effect prediction through structural pretraining and DMS-supervised LoRA fine-tuning. ProMut learns local structural environments with iRMB and VoxelRoI, then adapts the representations to experimental mutation rankings.

## Overview

![ProMut framework: structural preprocessing, pretraining and LoRA fine-tuning](docs/framework.png)

```text
Protein structures → 7-channel voxel grids → Structural pretraining
                                                      ↓
Experimental DMS data + pretrained backbone → LoRA fine-tuning
                                                      ↓
                                            Mutation effect ranking
```

[View the full-resolution PDF](docs/framework.pdf) · [Implementation notes](docs/整理说明.md)

The figure summarizes local preprocessing, structural pretraining, LoRA fine-tuning and downstream analyses. Results shown in the supplied figure were not recomputed in this repository update.

## Install

The code has been smoke-tested with Python 3.8.20 and PyTorch 2.4.1. Install PyTorch for your hardware; the following commands use the CPU build:

```bash
git clone https://github.com/AGI-FBHC/ProMut.git
cd ProMut
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For PDB preprocessing, also run `python -m pip install -r requirements-preprocess.txt`. PDB2PQR and FreeSASA are required at this stage. Use an ASCII-only working path if FreeSASA reports a path-encoding error. GPU training is recommended for the full datasets.

## Data

**Datasets and model weights are not hosted in this repository. To obtain them, email [luochenxi@stu.jiangnan.cn](mailto:luochenxi@stu.jiangnan.cn), specifying the files you need.**

| Resources available on request | Purpose |
|---|---|
| `data_csv/`, `promut2_dataset/` | Processed structure CSVs and input PDBs |
| `train_data/` including its NPZ files and original RAR archives | Prepared pretraining data and labels |
| `data_test_0.npz`, `label_test_0.npz`, `testset_TS50/`, `testset_T500/` | Structural evaluation data |
| `train_0613.npz`, `val_0613.npz` | Prepared DMS training and validation data |
| Pretrained model, `best_7093model.pth.rar` and its extracted contents | Structural model checkpoint resources |

For the commands below, obtain the **complete, loadable checkpoint file** and place it at `ProMut1/best_7093model.pth.tar`; an extracted folder is not a checkpoint file. DMS NPZ files belong in the project root. See [data formats](docs/DATA.md).

## 1. Workflow

Run all commands from the project root. **With the provided pretrained checkpoint and DMS NPZ files, skip directly to section 1.3.**

### 1.1 Local microenvironments

**Purpose:** encode residue-centered coordinates, charge and SASA as standardized local voxels to learn structural and physicochemical environments.

| Structure input | Role |
|---|---|
| `data_csv/*.csv` | Approximately 30,016 natural protein structure CSVs, primarily for pretraining |
| `promut2_csv/csv/*.csv` | Converted from `promut2_dataset/*.pdb` to represent DMS-associated proteins |

Convert training PDBs into structural features. Create and populate `pretrain_pdb/` with training structures only, keeping evaluation proteins separate:

```bash
python -c "import sys; sys.path.insert(0,'ProMut1'); from data_utils import process_pdb2csv; process_pdb2csv('pretrain_pdb','pretrain_structure')"
```

This writes CSVs to `pretrain_structure/csv/`. Set the following in `ProMut1/path_config.py`:

```python
DATA_CSV_DIR = os.path.join(PROJECT_ROOT, "pretrain_structure", "csv")
PRETRAIN_TRAIN_DATA_DIR = PRETRAIN_SHARD_DIR
```

Then build voxel grids and training shards:

```bash
python ProMut1/generate_full_sidechain_box_20A.py
python ProMut1/build_pretrain_shards.py
```

Outputs: `pretrain_npz/` → `pretrain_shards/data_0.npz`–`data_19.npz` and the matching `label_0.npz`–`label_19.npz`. Each residue is represented by a `7×20×20×20` grid. Confirm all 20 pairs exist before training.

Already have `data_csv/`? Skip PDB conversion and retain the default `DATA_CSV_DIR`. Already have `train_data/`? Skip preprocessing and set `PRETRAIN_TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, "train_data")` instead.

### 1.2 Structural pretraining

**Purpose:** predict the central wild-type amino acid from its local structure, starting from random initialization. Training uses 20-class cross-entropy; the `[N, 20]` outputs describe structural compatibility, not direct DMS function scores.

```bash
python ProMut1/train.py
```

Input: the configured pretraining shards; optional evaluation files are `data_test_0.npz` and `label_test_0.npz`. Defaults are 8 epochs and batch size 150, configured in `train.py`.

Output: `model_save/EMOCPD_VoxelRoI/train_epoch_*.pth.tar`; `best*model.pth.tar` is saved when periodic evaluation accuracy improves. Use an actual generated checkpoint in section 1.3 if training from scratch.

### 1.3 Mutation fine-tuning

**Purpose:** freeze backbone parameters and train LoRA adapters using a differentiable ranking loss to match experimental DMS rankings.

Inputs are `train_0613.npz` and `val_0613.npz`. Original labels come from `gym_single.csv` and `dms_single_fix.csv`, with protein identifiers, mutations such as `A11D`, and experimental scores. These CSVs do not directly replace the prepared NPZ files.

```bash
python ProMut1/rl_mutation_model.py --train_data train_0613.npz --val_data val_0613.npz --pretrained_model ProMut1/best_7093model.pth.tar --epochs 500 --batch_size 200 --output_dir model_save/ProMut_2
```

For the from-scratch route, replace `--pretrained_model` with the checkpoint from section 1.2. The entry point enables backbone parameter freezing, LoRA and augmentation by default.

Outputs: `model_save/ProMut_2/best_model.pth`, periodic `model_epoch_*.pth`, `final_model.pth`, and per-protein analysis in `analysis_outputs/`. Preserve previous outputs before repeating a run.

## 2. Experiments

### 2.1 TS50 / TS500 sequence recovery

**Purpose:** evaluate recovery of the central residue type from local structure. Top-1 accuracy is the fraction of correctly predicted amino acids.

Convert test PDBs into CSVs. The example uses TS50; TS500 is stored under `testset_T500/` in this project:

```bash
python -c "import sys; sys.path.insert(0,'ProMut1'); from data_utils import process_pdb2csv; process_pdb2csv('testset_TS50','testset_TS50_csv')"
```

Run prediction for one generated CSV, replacing `protein.csv` and the chain with actual values. This entry point builds voxels directly from CSV and does not require a test NPZ first.

```bash
python ProMut1/predict_emocpd.py --protein-csv testset_TS50_csv/csv/protein.csv --model ProMut1/best_7093model.pth.tar --chain A --topk 20 --out predictions/TS50_protein.csv
```

Outputs include `Actual AA`, `Pred AA 1` and candidate confidence scores. Repeat for all test proteins, then aggregate over evaluated residues:

```python
from pathlib import Path
import pandas as pd

results = pd.concat([pd.read_csv(p) for p in Path("predictions").glob("TS50_*.csv")])
accuracy = (results["Actual AA"] == results["Pred AA 1"]).mean()
print(f"Residue-weighted recovery: {accuracy:.4f}")
```

This is residue-weighted recovery; protein-averaged recovery requires a separate aggregation and should be labeled accordingly. Optionally add `--effect-out predictions/effects.csv` for pretrained structural log-ratios, `log P(mutant) − log P(wild type)`.

### 2.2 DMS mutation ranking

**Purpose:** assess agreement with experimental mutation rankings rather than wild-type classification.

During the fine-tuning command in section 1.3, `rl_mutation_model.py` evaluates `val_0613.npz` by protein each epoch. The core calculation is illustrated below (a fragment from the training loop):

```python
outputs = model(data)
loss = criterion2(outputs, mutation_labels, mutation_types, ranked_types)
correlation = -loss.item()
```

The implementation uses soft ranks for its correlation loss and writes per-protein records to `analysis_outputs/`. This training surrogate should not be equated with a separately computed standard hard-rank Spearman coefficient. Specify residue, protein and dataset aggregation when reporting results.

The lowest validation loss selects `best_model.pth`; training completion saves `final_model.pth`. A standalone fine-tuned inference entry point is not included, and these checkpoints cannot be loaded directly by `predict_emocpd.py`.

| Stage | Supervision | Main objective | Interpretation |
|---|---|---|---|
| Pretraining | Wild-type amino-acid identity | Cross-entropy / sequence recovery | Local structural compatibility |
| Fine-tuning | Experimental DMS scores and ranks | Differentiable ranking correlation loss | Mutation function ranking |

## Notes

- Public model names are `ProMutBackbone` and `ProMutPredictor`; `EMOCPD_VoxelRoI` and `HybridMutationPredictor` remain compatible aliases. Historical script names and output paths are retained.
- Fine-tuning requires prepared NPZ files with `data`, `positions`, `protein_names`, `mutation_labels`, `mutation_types` and `ranked_types`. The complete raw-DMS-to-NPZ preparation script is not included.
- `predict_emocpd.py` loads the **pretrained backbone**, not the fine-tuned predictor. Its scores are not the DMS fine-tuning results; a standalone fine-tuned inference entry point is not included.
- Smoke tests do not establish full reproduction of the reported experimental results.
