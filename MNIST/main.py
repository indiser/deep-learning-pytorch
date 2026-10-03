import torch
from torch import nn, save, load
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim import Adam


EPOCHES = 10
TRAIN = False
MODEL_PATH = 'model.pt'

def load_train_data(batch_size = 32):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root = "./train",
        download = True,
        train = True,
        transform = transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_test_data(batch_size = 32):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root = "./test",
        download = True,
        train = False,
        transform = transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def show_predictions(model, loader, n=12):
    model.eval()
    images, labels = next(iter(loader))
    with torch.no_grad():
        preds = model(images.to(device)).argmax(dim=1).cpu()
    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        color = "green" if preds[i] == labels[i] else "red"
        plt.title(f"pred {preds[i].item()} / true {labels[i].item()}", color=color)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

train_set = load_train_data()
test_set = load_test_data()

class ClassificationModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 32)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.ac2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        x = self.ac2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClassificationModel().to(device)
    optimizer = Adam(
        model.parameters(),
        lr=0.001,
        eps=1e-7,
        betas=(0.9, 0.999)
    )

    loss_func = nn.CrossEntropyLoss()

    if TRAIN:
        for epoch in range(1, EPOCHES + 1):
            for images, lables in train_set:
                images, lables = images.to(device), lables.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_func(outputs, lables)
                loss.backward()
                optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

        save(model.state_dict(), MODEL_PATH)

    else:
        model.load_state_dict(load(MODEL_PATH, map_location=device))

    acc = evaluate(model, test_set)
    print(f"Test accuracy: {acc:.2%}")

    show_predictions(model, test_set)