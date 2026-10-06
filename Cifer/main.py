import torch
import torch.nn as nn
from torch import save, load
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from PIL import Image

IS_TRAINED = True
MODEL_PATH = 'model.pt'
EPOCHS = 10
Image_PATH = "gettyimages-1500448395-612x612.jpg"

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

new_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_train_data(batch_size = 128):
    dataset = datasets.CIFAR10(
        root = "./train",
        download=True,
        train= True,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def load_test_data(batch_size = 128):
    dataset = datasets.CIFAR10(
        root = "./test",
        download=True,
        train= False,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

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

def predict_image(model, image_path):
    img = Image.open(image_path)
    img = new_transform(img)
    img = img.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        return classes[predicted]


class NuralNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NuralNet, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = load_train_data()
    test_set = load_test_data()

    model = NuralNet().to(device)

    if IS_TRAINED == False:
        loss_fc = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001, eps=1e-7, betas=(0.9, 0.999))

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for inputs, lables in train_set:
                inputs, lables = inputs.to(device), lables.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fc(outputs, lables)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch: {epoch}, Loss: {running_loss/len(train_set):.4f}")

        save(model.state_dict(), MODEL_PATH)
    else:
        model.load_state_dict(load(MODEL_PATH, map_location=device))
    
    acc = evaluate(model, test_set)
    print(f"Test accuracy: {acc:.2%}")

    print(f"Prediction: {predict_image(model, Image_PATH)}")