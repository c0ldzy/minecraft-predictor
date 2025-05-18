import os, io, torch, torch.nn as nn
from torchvision import transforms
from PIL import Image

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

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'mc_model.pth')
model = CNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(file_storage):
    img = Image.open(io.BytesIO(file_storage.read())).convert('RGB')
    x = tf(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)[0]
    
    return {
        'pig': probs[1].item(),
        'chicken': probs[0].item()
    }