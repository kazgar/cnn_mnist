import sys
import torch
from model import MNIST_CNN
from functions import predict_image, transform_image
from torchinfo import summary

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

def main():
    if len(sys.argv) != 3:
        raise Exception("Usage: python* <model> <image>")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cnn_model = MNIST_CNN(1, 20, 10).to(device)
    cnn_model.load_state_dict(torch.load(sys.argv[1]))
    prediction_idx = predict_image(cnn_model, sys.argv[2]).item()
    print(f"The number in the picture is {class_names[prediction_idx]}")

if __name__=="__main__":
    main()