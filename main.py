import sys
import torch
from model import MNIST_CNN
from functions import predict_image, transform_image, setup_model
from torchinfo import summary
import pygame 
import time

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
OPENSANS = "assets/fonts/OpenSans-Regular.ttf"
SMALL_FONT = pygame.font.Font(OPENSANS, 20)
LARGE_FONT = pygame.font.Font(OPENSANS, 40)
ROWS, COLS = 28, 28
OFFSET = 20
CELL_SIZE = 10
SIZE = WIDTH, HEIGHT = 1000, 800
handwriting = [[0] * COLS for _ in range(ROWS)]
classification = None


def main():
    if len(sys.argv) != 2:
        raise Exception("Usage: python main.py model")
    cnn_model = setup_model(sys.argv[1])

    screen = pygame.display.set_mode(SIZE)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        
        screen.fill(BLACK)

        click, _, _ = pygame.mouse.get_pressed()
        if click == 1:
            mouse = pygame.mouse.get_pos()
        else:
            mouse = None

        cells = []


if __name__=="__main__":
    main()