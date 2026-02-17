import pygame
import asyncio
import random
import instanceBob
import utilsBob

pygame.init()

LoopClock = pygame.time.Clock()

Surface = pygame.display.set_mode((500,500))
pygame.display.set_caption("Bob Evolution Simulator", "BES")

Bob = instanceBob.Bob(Surface, 400,235,20, (0,0,0))
Target = instanceBob.Target(Surface, 50, 215, 50, (255,0,0))

Step = 2

CurrentState = None

dx = None
dy = None

def predictionMapping(mI):
    if mI == 0: #Forward
        Bob.x -= Step
    if mI == 1: #Left
        Bob.y += Step
    if mI == 2: #Backward
        Bob.x += Step

#hidden layer: 4 neurons, :type List{weights:List{}, bias:int, result:int}
firstLayer = []
for i in range(4):
    firstLayer.append([[random.uniform(-0.1,0.1), random.uniform(-0.1,0.1)], random.uniform(-0.1,0.1), 0])

#output layer: 4 neurons, :type List{weights:List{}, bias:int, result:int}
outerLayer = []
for i in range(4):
    outerLayer.append([[random.uniform(-0.1,0.1), random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),random.uniform(-0.1,0.1)], random.uniform(-0.1,0.1), 0])


Alpha = 0.02 #learning rate
Gamma = 0.8 #its care for rewards
Epsilon = 0.95 #exploring rate 

Simulating = True

while Simulating:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            Simulating = False

    Surface.fill((255,255,255))

    Bob.draw()
    Target.draw()

    prediction = utilsBob.ForwardPass()



    pygame.display.update()

    LoopClock.tick(60) #60 fps