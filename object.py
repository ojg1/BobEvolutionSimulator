

import pygame
class Bob:
    def __init__(self, surface, x, y, sideLength, color : tuple):
        self.x = x
        self.y = y
        self.sideLength = sideLength
        self.color = color
        self.surface = surface

        

    def draw(self):
        return pygame.draw.rect(self.surface, (self.color), pygame.Rect(self.x, self.y, self.sideLength, self.sideLength))
    

class Target:
    def __init__(self, surface, x, y, sideLength, color : tuple):
        self.x = x
        self.y = y
        self.sideLength = sideLength
        self.color = color
        self.surface = surface

    def draw(self):
        return pygame.draw.rect(self.surface, self.color, pygame.Rect(self.x,self.y,self.sideLength,self.sideLength))
