import torch
from torch.utils.data import Dataset, TensorDataset

class MyDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        images = []
        if self.transforms is not None:
            for transform in self.transforms:
                images.append(transform(image))
        else:
            raise NotImplementedError

        return torch.stack(images), label
