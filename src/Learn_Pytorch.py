from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import os
import pickle

# train_dataset = CIFAR10(root="./data/raw/cifar10", train=True, download=True)
# test_dataset = CIFAR10(root="./data/raw/cifar10", train=False, download=True)
# image, label = train_dataset.__getitem__(2000)
# print(label)
# image.show()





##########################
# TẠO DATASET CỦA MÌNH
class MyDataset(Dataset):
    def __init__(self, root, train=True):
        if train:
            data_files = [os.path.join(root, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(root, "test_batch")]

        self.images = []
        self.labels = []
        for data_file in data_files:
            with open(data_file, "rb") as fo:
                batch = pickle.load(fo, encoding="bytes")
                self.images.extend(batch[b"data"])
                self.labels.extend(batch[b"labels"])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

if __name__ == "__main__":
    dataset = MyDataset(root="", train=True)
    image, label = dataset.__getitem__(2000)
    print(label)
    print(image.shape)