import os


PRO_MUT1_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PRO_MUT1_DIR, os.pardir))

DATA_CSV_DIR = os.path.join(PROJECT_ROOT, "data_csv")
PROMUT2_PDB_DIR = os.path.join(PROJECT_ROOT, "promut2_dataset")
PROMUT2_CSV_DIR = os.path.join(PROJECT_ROOT, "promut2_csv")
PRETRAIN_NPZ_DIR = os.path.join(PROJECT_ROOT, "pretrain_npz")
PRETRAIN_SHARD_DIR = os.path.join(PROJECT_ROOT, "pretrain_shards")
PRETRAIN_TRAIN_DATA_DIR = PRETRAIN_SHARD_DIR
PRETRAIN_TEST_DATA = os.path.join(PROJECT_ROOT, "data_test_0.npz")
PRETRAIN_TEST_LABEL = os.path.join(PROJECT_ROOT, "label_test_0.npz")
TRAIN_NPZ = os.path.join(PROJECT_ROOT, "train_0613.npz")
VAL_NPZ = os.path.join(PROJECT_ROOT, "val_0613.npz")
PRETRAINED_MODEL = os.path.join(PRO_MUT1_DIR, "best_7093model.pth.tar")
FINETUNE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_save", "ProMut_2")
PREDICTION_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "predictions")
ANALYSIS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis_outputs")
