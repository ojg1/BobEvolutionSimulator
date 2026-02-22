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
    elif mI == 1: #Left
        Bob.y += Step
    elif mI == 2: #Backward
        Bob.x += Step
    elif mI == 3: #Right
        Bob.y -= Step

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




    DistanceMagnitude = utilsBob.DistanceMagnitude(Target, Bob, 500, 500) 
    dx, dy = DistanceMagnitude[0], DistanceMagnitude[1] #GetCurrentState() -> (dx, dy)
    dxO, dyO = dx, dy
    #Forward pass, prediction [dir:index, val:int]
    Cprediction = utilsBob.ForwardPass(firstLayer, outerLayer, dx, dy)
    predictionMapping(Cprediction[0])

    #2nd Forward pass, prediction [dir:index, val:int] max, no step
    Mprediction = utilsBob.ForwardPass(firstLayer, outerLayer, dx, dy)
    DistanceMagnitude = utilsBob.DistanceMagnitude(Target, Bob, 500, 500) 
    dx, dy = DistanceMagnitude[0], DistanceMagnitude[1] #GetCurrentState() -> (dx, dy)

    #         may tweak scale to /500 since window is /500
    #         1 - sqrt(dx^2+dy^2)/100
    Reward = (1-utilsBob.RewardMagnitude(dx,dy))
    TDerror = (Reward + Gamma * Mprediction[1]) - Cprediction[1]
    
    #backprop, tweak weights on network
    Bprop = utilsBob.Backpropagate(oNeuronIndex=Cprediction[0], outerLayer=outerLayer, firstLayer=firstLayer, TDerror=TDerror, loss=(TDerror**2), Alpha=Alpha, dx=dxO, dy=dyO)
    firstLayer = Bprop[0]
    outerLayer = Bprop[1]


    pygame.display.update()

    LoopClock.tick(60) #60 fps