import os, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_dir = 'train/processed'
tf = transforms.Compose([
  transforms.Resize((128,128)),
  transforms.ToTensor(),
  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

ds = datasets.ImageFolder(data_dir, transform=tf)
dl = DataLoader(ds, batch_size=8, shuffle=True)
print('Classes:', ds.classes, 'Samples:', len(ds))

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
      nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    )
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(32*32*32,128), nn.ReLU(),
      nn.Linear(128,2)
    )
  def forward(self,x): return self.fc(self.conv(x))

model = CNN()
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
  correct = total = 0
  for imgs, labels in dl:
    out = model(imgs)
    loss = loss_fn(out, labels)
    opt.zero_grad(); loss.backward(); opt.step()
    preds = out.argmax(1)
    total += labels.size(0)
    correct += (preds==labels).sum().item()
  print(f'Epoch {epoch+1}/5  acc={100*correct/total:.1f}%')

os.makedirs('model', exist_ok=True)
torch.save(model, 'model/mc_model.pth')
print('Saved model → model/mc_model.pth')
