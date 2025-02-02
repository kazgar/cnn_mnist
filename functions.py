import torch 
import torchvision
from torchvision import transforms
from pathlib import Path
from model import MNIST_CNN

def transform_image(image_path):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
    ])

    image_path = Path(image_path)
    image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    image /= 255
    image = transform(image)
    if image.ndimension() == 2:  # If image is HxW, add channel dim
        image = image.unsqueeze(0)
    image = image.unsqueeze(dim=0).to(device)
    return image

def predict_image(model, image_path):
    image = transform_image(image_path)
    model.eval()
    with torch.inference_mode():
        pred = model(image)
        pred = torch.argmax(torch.softmax(pred, dim=1))
        return pred

def setup_model(model_path):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cnn_model = MNIST_CNN(1, 20, 10).to(device)
    cnn_model.load_state_dict(torch.load(model_path))
    return cnn_model
