import torch
from torch.utils.data import Dataset


class MoaDataset(Dataset):
    def __init__(self, data, targets, feats_idx, mode="train"):
        """TabNet dataset.

        Args:
            data: ndarray (N, original_cols)
            target: ndarray (N, 1)
            feats_idx: list of index
            mode : str train or test
        Returns:
            X,
        """
        self.mode = mode
        self.feats = feats_idx
        self.data = data[:, feats_idx]
        if mode == "train":
            self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "train":
            return {
                "X": torch.FloatTensor(self.data[idx]),
                "y": torch.FloatTensor(self.targets[idx]),
            }
        elif self.mode == "test":
            return {"X": torch.FloatTensor(self.data[idx]), "y": 0}
