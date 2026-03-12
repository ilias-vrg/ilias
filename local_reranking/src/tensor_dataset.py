import os
import tempfile

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler, SequentialSampler, DataLoader


class TensorFileDataset(Dataset):
    def __init__(self, feature_files, desc_num=None):
        super().__init__()
        self.feature_files = feature_files
        self.desc_num = desc_num
        self.local_storage = self.v_file_handle = dtype = max_feature_shape = None

        images_per_hdf5, index_per_hdf5, file_shapes = [], [], []

        for file in feature_files:
            with h5py.File(file, "r") as f:
                current_shape = f["features"].shape
                current_feature_dims = list(current_shape[1:])

                if dtype is None:
                    dtype = f["features"].dtype
                    max_feature_shape = current_feature_dims
                else:
                    assert dtype == f["features"].dtype
                    assert max_feature_shape[-1] == current_shape[-1]
                    max_feature_shape[0] = max(max_feature_shape[0], current_shape[1])

                file_shapes.append(current_shape)
                images_per_hdf5.append(current_shape[0])
                index_per_hdf5.append(f["index"][:])

        self.num_samples = sum(images_per_hdf5)
        self.index = np.concatenate(index_per_hdf5, axis=0).astype(str)

        if self.num_samples < 1e5:
            self.local_storage = np.zeros(
                (self.num_samples, *max_feature_shape), dtype=dtype
            )

            current_idx = 0
            for file, shape in zip(feature_files, file_shapes):
                with h5py.File(file, "r") as f:
                    data = f["features"][:]  # Load data into memory
                    num_imgs, width = shape[0], shape[1]
                    self.local_storage[
                        current_idx : current_idx + num_imgs, 0:width
                    ] = data
                current_idx += num_imgs
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.v_file_path = os.path.join(self.temp_dir.name, "shared_v_map.hdf5")

            with h5py.File(self.v_file_path, "w") as f:
                layout_shape = (self.num_samples, *max_feature_shape)
                layout = h5py.VirtualLayout(shape=layout_shape, dtype=dtype)

                total = 0
                for chunk_file, num_images, current_shape in zip(
                    self.feature_files, images_per_hdf5, file_shapes
                ):
                    vsource = h5py.VirtualSource(
                        chunk_file, "features", shape=current_shape, dtype=dtype
                    )
                    width = current_shape[1]
                    layout[total : total + num_images, 0:width] = vsource
                    total += num_images
                f.create_virtual_dataset("features", layout, fillvalue=0)

    def _init_storage(self):
        self.v_file_handle = h5py.File(self.v_file_path, "r", swmr=True)
        self.local_storage = self.v_file_handle["features"]

    def __len__(self):
        return self.num_samples

    def __del__(self):
        if hasattr(self, "v_file_handle") and self.v_file_handle:
            self.v_file_handle.close()
        if hasattr(self, "temp_dir") and self.temp_dir:
            self.temp_dir.cleanup()

    def __getitem__(self, batch_index):
        if self.local_storage is None:
            self._init_storage()
        idx = np.sort(np.unique(batch_index)).tolist()
        local_feat = self.local_storage[idx, : self.desc_num]

        local_feat = local_feat[[idx.index(i) for i in batch_index]]
        masks = local_feat.sum(axis=-1) == 0

        if local_feat.dtype == np.uint8:
            local_feat = np.unpackbits(local_feat, axis=-1).astype(float)
            local_feat = 2 * local_feat - 1

        local_feat = torch.from_numpy(local_feat).float()
        masks = torch.from_numpy(masks)

        return (local_feat, masks), batch_index


def get_test_loader(feature_files, batch_size, num_workers=8, desc_num=None):
    test_set = TensorFileDataset(feature_files, desc_num)

    sampler = BatchSampler(
        SequentialSampler(test_set), batch_size=batch_size, drop_last=False
    )
    loader = DataLoader(
        test_set,
        shuffle=False,
        sampler=sampler,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: x[0],
    )

    return loader
