import sys
import torch
from model import MNIST_CNN
from functions import predict_image, transform_image, setup_model
from torchinfo import summary
import random
import time
import pygame

WIDTH, HEIGHT = 600, 600

pygame.init()

def main():
    if len(sys.argv) != 2:
        raise Exception("Usage: python main.py model")
    cnn_model = setup_model(sys.argv[1])

    ### PYGAME

    clock = pygame.time.Clock()
    font_little = pygame.font.SysFont("Roboto-Regular.ttf", 30) 
    font_small = pygame.font.SysFont("Roboto-Regular.ttf", 50)
    font_big = pygame.font.SysFont("Roboto-Regular.ttf", 400)
    screen = pygame.display.set_mode((1200,600))
    screen.fill("white")
    canvas = pygame.Surface((WIDTH, HEIGHT))
    canvas.fill("black")
    drawing = False
    running = True
    text = font_small.render("CNN thinks your number is...", True, "black")
    screen.blit(text, (660, HEIGHT/2 - 200))
    info = font_little.render("ENTER = predict number | SPACE = clear canvas", True, "black")
    screen.blit(info, (660, HEIGHT/2 - 160))
    pygame.display.set_caption("INTERACTIVE UI FOR CNN PREDICTION ON MNIST")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not drawing:
                drawing = True
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    pygame.draw.circle(canvas, "white", event.pos, 15)
            elif event.type == pygame.MOUSEBUTTONDOWN and drawing:
                drawing = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    pygame.image.save(canvas, "to_predict.jpeg")
                    prediction = predict_image(cnn_model, "to_predict.jpeg")
                    text_predict = font_big.render(str(prediction.item()), True, "black")
                    screen.blit(text_predict, (820, HEIGHT/2 - 100))
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    screen.fill("white")
                    canvas.fill("black")
                    screen.blit(text, (660, HEIGHT/2 - 200))
                    screen.blit(info, (660, HEIGHT/2 - 160))

        
        screen.blit(canvas, (0, 0))
        pygame.display.flip()

        clock.tick(60)
    
    pygame.quit()


if __name__=="__main__":
    main()
