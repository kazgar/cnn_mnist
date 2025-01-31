import torch 
import torchvision
from torchvision import transforms
from pathlib import Path

def transform_image(image_path):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    transform = transforms.Compose([
        transforms.Resize((28, 28))
    ])

    image_path = Path(image_path)
    image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    image /= 255
    image = transform(image).unsqueeze(dim=0).to(device)
    return image

def predict_image(model, image_path):
    image = transform_image(image_path)
    model.eval()
    with torch.inference_mode():
        pred = model(image)
        pred = torch.argmax(torch.softmax(pred, dim=1))
        return pred