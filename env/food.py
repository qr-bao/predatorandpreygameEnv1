import pygame
import env.constants as constants

class Food:
    def __init__(self, x, y, size):
        self.rect = pygame.Rect(x, y, size, size)
        self.color = constants.FOOD_COLOR
        self.type = "food"

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
