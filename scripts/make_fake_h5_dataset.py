import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser('Create a small h5-only PRET smoke-test dataset')
    parser.add_argument('--out', default='data/FAKEH5/h5')
    parser.add_argument('--slides', type=int, default=20)
    parser.add_argument('--patches', type=int, default=96)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1024)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    grid_w = int(np.ceil(np.sqrt(args.patches)))
    for slide_idx in range(args.slides):
        label = slide_idx % 2
        center = np.zeros(args.dim, dtype=np.float32)
        center[label] = 1.0

        features = []
        coordinates = []
        for patch_idx in range(args.patches):
            x = patch_idx % grid_w
            y = patch_idx // grid_w
            feat = center + rng.normal(0, 0.06, args.dim).astype(np.float32)
            feat = feat / np.linalg.norm(feat)
            features.append(feat)
            coordinates.append([x, y])

        h5_path = out_dir / f'fake_h5_slide_{slide_idx:02d}.h5'
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('features', data=np.stack(features, 0))
            f.create_dataset('coordinates', data=np.asarray(coordinates, dtype=np.int32))

    print(f'Wrote {args.slides} h5 files to {out_dir}')
    print('No labels were written; PRET will assign deterministic pseudo labels for smoke tests.')


if __name__ == '__main__':
    main()
