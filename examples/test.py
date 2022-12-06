import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME


def print_title(s, data):
    print('='*20, s, '='*20)
    print(data)


if __name__ == '__main__':
    origin_pc1 = 100 * np.random.uniform(0, 1, (10, 3))
    feat1 = np.ones((10, 3), dtype=np.float32)
    origin_pc2 = 100 * np.random.uniform(0, 1, (6, 3))
    feat2 = np.ones((6, 3), dtype=np.float32)
    print_title('origin_pc1', origin_pc1)
    print_title('origin_pc2', origin_pc2)

    coords, feats = ME.utils.sparse_collate([origin_pc1, origin_pc2], [feat1, feat2])
    print_title('coords', coords)
    print_title('feats', feats)
    input = ME.SparseTensor(feats, coordinates=coords)
    print_title('input', input)
