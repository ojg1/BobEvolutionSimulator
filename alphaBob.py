import pygame
import instanceBob
# import numpy
import math
import time
import asyncio
import random


Cock = pygame.time.Clock()

ScreenWidth = 500
ScreenHeight = 500
                # r g b
BackgroundColor = (255,255,255)

Screen = pygame.display.set_mode((ScreenWidth, ScreenHeight))

Bob = instanceBob.Bob(Screen, 400,250,20, (0,0,0))
Target = instanceBob.Target(Screen, 50, 225, 50, (255,0,0))

RunningLoop = True

steppingStone = 5

# Epsilon = 1
Alpha = 0.01
Gamma = 0.85
Epsilon = 0.95

offsetFromTargetX = None #inputs
offsetFromTargetY = None #inputs
Rewards = []
TimeHesitationThreshold = 15
timepassed = False
MaxDistance = None
currentMove = 0
nextState = None
Inputs = [offsetFromTargetX, offsetFromTargetY]

firstLayerBias = 0.7
firstLayer = [
    [[0,0],firstLayerBias,0],
    [[0,0],firstLayerBias,0],
    [[0,0],firstLayerBias,0],
    [[0,0],firstLayerBias,0]
]
# format: 4 weights, bias, result

outputLayerBias = 0.7
outputLayer = [
    [[0,0,0,0],outputLayerBias,0], # 0 = fwd
    [[0,0,0,0],outputLayerBias,0], # 1 =
    [[0,0,0,0],outputLayerBias,0],
    [[0,0,0,0],outputLayerBias,0]
]


async def startClock():
    time.sleep(TimeHesitationThreshold)
    timepassed = True


correspondents = {"Forward":0, "Left":1, "Backward":2, "Right":3}


def epsilonGreedy(oLvalues):
    rand = random.uniform(0, 1)
    if rand > Epsilon:
        # Exploit: choose the best predicted move
        max_value = oLvalues[0]
        max_index = 0
        for i in range(1, len(oLvalues)):
            if oLvalues[i] > max_value:
                max_value = oLvalues[i]
                max_index = i
        choice = PossibleDirections[max_index]
    else:
        # Explore: choose a random move
        choice = random.choice(PossibleDirections)
        max_index = correspondents[choice]
    return (choice, oLvalues[max_index], max_index)

def calculateNextPosition():

    for i in range(len(firstLayer)):
        firstLayer[i][2] = offsetFromTargetX*firstLayer[i][0][0] + offsetFromTargetY*firstLayer[i][0][1] + firstLayer[i][1]

    oLvalues = []   

    for i in range(len(outputLayer)):
        weightsNeuron = []
        for j in range(len(firstLayer)-1):
            weightsNeuron.append(firstLayer[j][2]*outputLayer[i][0][j])
        outputLayer[i][2] = sum(weightsNeuron)+outputLayer[i][1]
        oLvalues.append(outputLayer[i][2])


    return epsilonGreedy(oLvalues)


    # maxIndex = max(oLvalues)


    # print(oLvalues)

    # oLvalues.index(maxIndex)

    # # Check if all outputs are equal
    # if oLvalues[0] == oLvalues[1] == oLvalues[2] == oLvalues[3]:
    #     choice = random.choice(PossibleDirections)
    #     return (choice, oLvalues[correspondents[choice]], correspondents[choice]) 

    # # Find max output
    # max_value = oLvalues[0]
    # max_index = 0
    # for i in range(1, len(oLvalues)):
    #     if oLvalues[i] > max_value:
    #         max_value = oLvalues[i]
    #         max_index = i

    # choice = PossibleDirections[max_index]
    # print(choice)

    # # Keep Bob on screen
    # if Bob.x > 500 or Bob.x < 0:
    #     Bob.x = abs(Bob.x % 500)
    # if Bob.y > 500 or Bob.y < 0:
    #     Bob.y = abs(Bob.y % 500)

    # return (choice, oLvalues[max_index], max_index)
    


def ActivationFunctionRectifiedLinearUnit(value):
    D = 0

    if value > 0:
        D = 1

    return (max(0, value), D)

def Backpropagate(error: float, index_of_output: int):
    """
    Backpropagation for a small 2-layer network.
    error: scalar error at the output neuron
    index_of_output: which output neuron to update
    Uses global Alpha, firstLayer, outputLayer, offsetFromTargetX/Y
    """

    # ---- Update Output Layer ----
    output_weights, output_bias = outputLayer[index_of_output]

    for w_idx in range(len(output_weights)):
        # hidden neuron output (ReLU)
        hidden_output = ActivationFunctionRectifiedLinearUnit(firstLayer[w_idx][2])[0]
        # weight update
        output_weights[w_idx] += Alpha * error * hidden_output

    # update output bias
    output_bias += Alpha * error
    outputLayer[index_of_output][1] = output_bias

    # ---- Update Hidden Layer ----
    for h_idx, hidden_neuron in enumerate(firstLayer):
        hidden_z = hidden_neuron[2]
        relu_derivative = ActivationFunctionRectifiedLinearUnit(hidden_z)[1]

        # blame is error propagated back through the output weight
        blame = error * output_weights[h_idx] * relu_derivative

        # update hidden weights (x and y)
        hidden_neuron[0][0] += Alpha * blame * offsetFromTargetX
        hidden_neuron[0][1] += Alpha * blame * offsetFromTargetY

        # update hidden bias
        hidden_neuron[1] += Alpha * blame



def calculateOffsets():
    global offsetFromTargetX
    global offsetFromTargetY
    # ??? why is it printing the target's x and bob's y testing purposes
    offsetFromTargetX = (Target.x - Bob.x)/500
    offsetFromTargetY = (Target.y - Bob.y)/500
    return 
def calculateMagnitude():
    Magnitude = math.sqrt((offsetFromTargetX**2)+(offsetFromTargetY**2))
    return Magnitude

# rewards = magnitude*147
    



PossibleDirections = ["Forward", "Backward", "Left", "Right"]

def PossibleDirectionMapping(action):
    if action == "Forward":
        Bob.x-= steppingStone
    elif action == "Left":
        Bob.y+=steppingStone
    elif action == "Backward":
        Bob.x+= steppingStone
    elif action == "Right":
        Bob.y -= steppingStone





while RunningLoop:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            RunningLoop = False

    Screen.fill(BackgroundColor)
    Bob.draw()
    Target.draw()

    calculateOffsets()

    predictedMove = calculateNextPosition()
    #print(predictedMove)
    PossibleDirectionMapping(predictedMove[0])
    magnitude = calculateMagnitude()
    reward = max(0, 1 - magnitude)  # now always between 0 and 1
    error = reward + Gamma * predictedMove[1] - currentMove

    currentMove = predictedMove[1] #int

    Backpropagate(error, predictedMove[2])
    print(firstLayer)
    print(outputLayer)

    Epsilon *= 0.995

    # calculateOffsets()

    # nextStateInputs = [offsetFromTargetX, offsetFromTargetY]

    Cock.tick(60)

    pygame.display.update()
            

