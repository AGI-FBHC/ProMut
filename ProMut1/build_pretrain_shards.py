import os

import numpy as np

from atom_res_dict import label_res_dict
from path_config import PRETRAIN_NPZ_DIR, PRETRAIN_SHARD_DIR


def build_pretrain_shards(source_dir=PRETRAIN_NPZ_DIR, output_dir=PRETRAIN_SHARD_DIR):
    os.makedirs(output_dir, exist_ok=True)

    for label, residue in label_res_dict.items():
        files = [
            os.path.join(source_dir, name)
            for name in os.listdir(source_dir)
            if name.startswith(residue + "_") and name.lower().endswith(".npz")
        ]
        files.sort()

        if not files:
            print(f"skip {residue}: no source npz files")
            continue

        arrays = [np.load(file)["arr"] for file in files]
        data = np.concatenate(arrays, axis=0)
        labels = np.full((data.shape[0],), label, dtype=np.int64)

        data_file = os.path.join(output_dir, f"data_{label}.npz")
        label_file = os.path.join(output_dir, f"label_{label}.npz")
        np.savez_compressed(data_file, data)
        np.savez_compressed(label_file, labels)
        print(f"{residue}: {data.shape} -> {data_file}")


if __name__ == "__main__":
    build_pretrain_shards()

