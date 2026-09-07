import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(3 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)


if __name__ == "__main__":
    num_epochs = 10
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = CIFAR10(
        root=r"D:\AI_LAB\DL_LAB\data\raw",
        train=True,
        download=False,
        transform=transform,
    )
    test_dataset = CIFAR10(
        root=r"D:\AI_LAB\DL_LAB\data\raw",
        train=False,
        download=False,
        transform=transform,
    )

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False)

    for images, labels in train_loader:
        print("images shape: {}".format(images.shape))
        print("labels shape: {}".format(labels.shape))
        break

    model = SimpleNeuralNetwork(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inter, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss_value = criterion(outputs, labels)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            running_loss += loss_value.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set: {100 * correct / total:.2f}%")

