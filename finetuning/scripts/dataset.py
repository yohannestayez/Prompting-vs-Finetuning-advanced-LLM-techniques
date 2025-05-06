from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            key: val[idx] for key, val in self.encodings.items()
        } | {"labels": self.labels[idx]}

    def __len__(self):
        return len(self.labels)
