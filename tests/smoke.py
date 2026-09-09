"""CPU checks; optional checkpoint must come from a trusted source."""
import argparse
import ast
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ProMut1'))

import numpy as np
import torch
from model_promut1 import EMOCPD_VoxelRoI, EMOCPD_NoIRMB, EMOCPD_NoMHSA, EMOCPD_NoIRMB_NoMHSA
from build_pretrain_shards import build_pretrain_shards
from datasets import NpzDataset
from atom_res_dict import label_res_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    for path in ROOT.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    x = torch.randn(2, 7, 20, 20, 20)
    for cls in (EMOCPD_VoxelRoI, EMOCPD_NoIRMB, EMOCPD_NoMHSA, EMOCPD_NoIRMB_NoMHSA):
        model = cls().eval()
        with torch.no_grad():
            result = model(x)
        assert result.shape == (2, 20)
        assert torch.isfinite(result).all()
        print(cls.__name__, 'OK')

    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / 'source'
        target = Path(temporary) / 'shards'
        source.mkdir()
        data = np.ones((2, 7, 20, 20, 20), dtype=np.float32)
        for label, residue in label_res_dict.items():
            np.savez_compressed(source / (residue + '_0.npz'), arr=data * label)
        build_pretrain_shards(str(source), str(target))
        for label in label_res_dict:
            dataset = NpzDataset(target / ('data_%d.npz' % label), target / ('label_%d.npz' % label))
            assert len(dataset) == 2
            assert dataset[0][1] == label
            np.testing.assert_array_equal(dataset[0][0], data[0] * label)
    print('Voxel shard -> training dataset round trip OK')

    if args.checkpoint:
        from rl_mutation_model import HybridMutationPredictor
        model = HybridMutationPredictor(str(args.checkpoint.resolve()), hidden_dim=640, lora_rank=16, lora_alpha=32)
        model.train()
        model(x).sum().backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for n, p in model.named_parameters() if 'lora_B' in n)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        print('Checkpoint load and LoRA gradients OK')


if __name__ == '__main__':
    main()
