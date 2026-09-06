from torch.utils.data import Dataset
import os
import pickle
import torch
import numpy as np
import cv2
from PIL import Image
# =======================================
# ================ TENSOR ===============
# =======================================
# a = torch.tensor(3)
# b = torch.tensor([1, 2, 3])
# c = torch.tensor([[1, 2, 3], [4, 5, 6]])
# d = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

# tensor_list = torch.tensor([[[1, 2, 3],
#                            [4, 5, 6],
#                             [7, 8, 9]]])
# print(tensor_list.shape)
# print(tensor_list.ndim)
# print(tensor_list.dtype)
# print(tensor_list.device)

# ================================================
# ================ Pytorch: Image to tensor ======
# ================================================
# # CASE 1: USE OPENCV
# import cv2
# import torch
# # image = cv2.imread(r'D:\AI_LAB\DL_LAB\data\raw\images\effen.jpg')
# # cv2.imshow("image", image)
# image = torch.from_numpy(image)
# print("image shape: {}".format(image.shape))
# print("Image number of dimensions: {}".format(image.ndim))
# cv2.waitKey(0)

# # CASE 2: USE PIL
# from PIL import Image
# from torchvision.transforms import ToTensor
# image = Image.open(r'D:\AI_LAB\DL_LAB\data\raw\images\effen.jpg')
# image.show()
# transform = ToTensor()
# image = transform(image)
# print("image shape: {}".format(image.shape))
# print("Image number of dimensions: {}".format(image.ndim))

# ================================================
# ================ Pytorch: Built-in datasets ====
# ================================================
# dataset = torchvision.datasets.Imagenet(root="data", split="train", download=True)
# dataloaders = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# ================================================
# ================ Pytorch: Dataset ==============
# ================================================
# CASE1: USE TORCHVISION DATASET (được định nghĩa sẵn)
# from torchvision.datasets import CIFAR10
# train_dataset = CIFAR10(root="data", train=True, download=True)
# test_dataset = CIFAR10(root="data", train=False, download=True)
# index = 2000
# image, label = train_dataset.__getitem__(index)
# image.show()
# print(image.size)
# print(image.label)
# print(train_dataset.classes)
# print(train_dataset.class_to_idx)

# # CASE 2: USE IMAGE FOLDER DATASET (dataset từ folder hình ảnh)
# from torchvision.datasets import ImageFolder


# CASE 3: DEFINE MY OWN DATASET (dataset tự định nghĩa)
# Kế thừa class torch.utils.data.Dataset và định nghĩa lại 3 phương thức: __init__, __len__, __getitem__
# class MyDataset(Dataset):
#     def __init__(self, root, train=True):
#         if train:
#             data_files = [os.path.join(root, "data_batch_{}".format(i)) for i in range(1, 6)]
#         else:
#             data_files = [os.path.join(root, "test_batch")]

#         self.images = []
#         self.labels = []
#         for data_file in data_files:
#             with open(data_file, "rb") as fo:
#                 batch = pickle.load(fo, encoding="bytes")
#                 self.images.extend(batch[b"data"])
#                 self.labels.extend(batch[b"labels"])

#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, index):
#         image = self.images[index]
#         label = self.labels[index]
#         return image, label

# if __name__ == "__main__":
#     dataset = MyDataset(root=r"D:\AI_LAB\DL_LAB\data\raw\cifar-10-batches-py", train=True)
#     image, label = dataset.__getitem__(234)
#     print("before", image.shape)
#     image = np.reshape(image, (3, 32, 32)) 
#     image = np.transpose(image, (1, 2, 0)) # chuyển từ (C, H, W) sang (H, W, C)
#     print("after", image.shape)
#     print(label)
#     print(image.shape)
#     cv2.imshow("image", cv2.resize(image, (320, 320)))
#     cv2.waitKey(0)


# ================================================
# ================ Pytorch: Dataset ==============
# ================================================
# class Animals_dataset(Dataset):
#     def __init__(self, root, train=True):
#         self.root = root
#         if train: 
#             mode = "train"
#         else: 
#             mode = "test"
#         self.root = os.path.join(root, mode)
#         print(os.listdir(self.root))
#         self.categories = ['cane', 
#                            'cavallo', 
#                            'elefante', 
#                            'farfalla', 
#                            'gallina', 
#                            'gatto', 
#                            'mucca', 
#                            'pecora', 
#                            'ragno', 
#                            'scoiattolo']
        
#         self.images_path = []
#         self.labels = []
#         for category in self.categories:
#             category_path = os.path.join(self.root, category)
#             # print("category_path: {}".format(category_path))
#             for image_name in os.listdir(category_path):
#                 # print("image_name: {}".format(image_name))
#                 image_path = os.path.join(category_path, image_name)
#                 image = cv2.imread(image_path)
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 image = cv2.resize(image, (224, 224))
#                 self.images_path.append(image)
#                 self.labels.append(self.categories.index(category))
                
#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, index):
#         image = self.images_path[index]
#         label = self.labels[index]
#         return image, label

# if __name__ == "__main__":
#     root = r"D:\AI_LAB\DL_LAB\data\raw\animals"
#     dataset = Animals_dataset(root=root, train=True)
#     print(dataset.root)
#     image, label = dataset.__getitem__(234)
#     print("image shape:", image.shape)
#     print("label:", label)
#     cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# ================================================
# ===== Pytorch: DATA LOADER =====================
# ================================================
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
if __name__ == "__main__":
    transform = Compose([Resize((32, 32)), 
                        ToTensor(), 
                        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    epochs = 10
    train_dataset = CIFAR10(root=r"D:\AI_LAB\DL_LAB\data\raw", 
                            train=True, 
                            download=False,
                            transform= transform)
    
    image, label = train_dataset.__getitem__(1234)
    training_loader = DataLoader(dataset= train_dataset, 
                                 batch_size=32, 
                                 shuffle=True,
                                 drop_last=True, 
                                 num_workers=4)
    for epoch in range(epochs):
        for images, labels in training_loader:
            print("images shape: {}".format(images.shape))
            print("labels shape: {}".format(labels.shape))
            break