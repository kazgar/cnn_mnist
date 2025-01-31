import sys
from pathlib import Path
from model import MNIST_CNN

class_names = {0: "Zero",
               1: "One",
               2: "Two",
               3: "Three",
               4: "Four",
               5: "Five",
               6: "Six",
               7: "Seven",
               8: "Eight",
               9: "Nine",
               10: "Ten"}


def transform_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((28, 28))
    ])
    
    image_path = Path(image_path)
    image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    image /= 255
    image = transform(image)
    image = image.unsqueeze(dim=0).to(device)
    return image

def predict_image(model, image):
    model.eval()
    with torch.inference_mode():
        pred = model(image)
        pred = torch.argmax(torch.softmax(image, dim=1), dim=1)
    return class_names[pred]

def main():
    if len(sys.argv) != 3:
        raise Exception("Usage: python* <model> <image>")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cnn_model = MNIST_CNN(1, 10, 10).to(device)
    cnn_model.load_state_dict(torch.load(sys.argv[1]))


if __name__=="__main__":
    main()