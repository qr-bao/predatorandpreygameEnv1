import pygame

class Obstacle:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = (100, 100, 100)  # 灰色

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
