import os, io, torch
from torchvision import transforms
from PIL import Image

tf = transforms.Compose([
  transforms.Resize((128,128)),
  transforms.ToTensor(),
  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

MODEL_PATH = os.path.join(os.path.dirname(__file__),'mc_model.pth')
model = torch.load(MODEL_PATH, map_location='cpu')
model.eval()

def predict_image(file_storage):
    img = Image.open(io.BytesIO(file_storage.read())).convert('RGB')
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)[0]
        probs = torch.softmax(out, dim=0).tolist()
    return {'pig': probs[0], 'chicken': probs[1]}
