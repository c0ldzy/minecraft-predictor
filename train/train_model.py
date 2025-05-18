import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

base_tf = transforms.Compose([
    transforms.Resize((128, 128))  # Only resizing
])

train_tf = transforms.Compose([
    transforms.RandomAffine(
        degrees=30,
        translate=(0.1, 0.1),
        scale=(0.8, 1.2),
        shear=10
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

base_ds = datasets.ImageFolder('train/processed', transform=base_tf)
print(f"Base samples: {len(base_ds)}  Classes: {base_ds.classes}")

class AugmentedDataset(Dataset):
    def __init__(self, ds, times=10, transform=None):
        self.ds = ds
        self.times = times
        self.transform = transform

    def __len__(self):
        return len(self.ds) * self.times

    def __getitem__(self, idx):
        img, label = self.ds[idx % len(self.ds)]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

aug_ds = AugmentedDataset(base_ds, times=10, transform=train_tf)
print(f"Unique augmented samples: {len(aug_ds)}")

loader = DataLoader(
    aug_ds,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16,256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256,2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
loss_fn = nn.CrossEntropyLoss()

epochs = 20
for epoch in range(1, epochs+1):
    model.train()
    running_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    avg_loss = running_loss / total
    acc = 100 * correct / total
    print(f"Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}  Acc: {acc:.1f}%")

# Save model weights
os.makedirs('model', exist_ok=True)
torch.save(model.state_dict(), 'model/mc_model.pth')
print("Model saved to model/mc_model.pth")