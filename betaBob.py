import pygame
import asyncio
import random
import instanceBob
import utilsBob

pygame.init()

LoopClock = pygame.time.Clock

Surface = pygame.display.set_mode((500,500))
pygame.display.set_caption("Bob Evolution Simulator", "BES")

Bob = instanceBob.Bob(Surface, 400,250,20, (0,0,0))
Target = instanceBob.Target(Surface, 50, 225, 50, (255,0,0))

Alpha = 0.02 #learning rate
Gamma = 0.8 #its care for rewards
Epsilon = 0.95 #exploring rate 

Simulating = True

while Simulating:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            Simulating = False

    Surface.fill(255,255,255)



    

    pygame.display.update()

    LoopClock.tick(60) #60 fps