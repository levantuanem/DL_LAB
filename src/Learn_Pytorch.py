from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import os
import pickle
import torch
# =======================================
# ================ TENSOR ===============
# =======================================
a = torch.tensor(3)
b = torch.tensor([1, 2, 3])
c = torch.tensor([[1, 2, 3], [4, 5, 6]])
d = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

tensor_list = torch.tensor([[[1, 2, 3],
                           [4, 5, 6],
                            [7, 8, 9]]])
print(tensor_list.shape)
print(tensor_list.ndim)
print(tensor_list.dtype)
print(tensor_list.device)

# ================================================
# ================ Pytorch: Image to tensor ======
# ================================================
# CASE 1: USE OPENCV
import cv2
import torch
image = cv2.imread(r'D:\AI_LAB\DL_LAB\data\raw\images\effen.jpg')
cv2.imshow("image", image)
image = torch.from_numpy(image)
print("image shape: {}".format(image.shape))
print("Image number of dimensions: {}".format(image.ndim))
cv2.waitKey(0)

# CASE 2: USE PIL
from PIL import Image
from torchvision.transforms import ToTensor
image = Image.open(r'D:\AI_LAB\DL_LAB\data\raw\images\effen.jpg')
image.show()
transform = ToTensor()
image = transform(image)
print("image shape: {}".format(image.shape))
print("Image number of dimensions: {}".format(image.ndim))


# ================================================
# ================ Pytorch: Dataset ==============
# ================================================
# CASE1: USE TORCHVISION DATASET (được định nghĩa sẵn)
from torchvision.datasets import CIFAR10
train_dataset = CIFAR10(root="data", train=True, download=True)
test_dataset = CIFAR10(root="data", train=False, download=True)
index = 2000
image, label = train_dataset.__getitem__(index)
image.show()
print(image.size)
print(image.label)
print(train_dataset.classes)
print(train_dataset.class_to_idx)

# CASE 2: USE IMAGE FOLDER DATASET (dataset từ folder hình ảnh)
from torchvision.datasets import ImageFolder


# CASE 3: DEFINE MY OWN DATASET (dataset tự định nghĩa)
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