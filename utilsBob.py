import math
import random

def DistanceMagnitude(Target, Bob, ScreenWidth, ScreenHeight):
    dx = (Target.x-Bob.x)/ScreenWidth
    dy = (Target.y-Bob.y)/ScreenHeight
    return (dx, dy)    

def RewardMagnitude(dx, dy):
    Magnitude = math.sqrt((dx**2)+(dy**2))/100
    return Magnitude


def RectifiedLinearUnit(z):
    d = 0
    if z > 0:
        d = 1
    return (max(0, z), d)


def ForwardPass(firstLayer : list, outerLayer : list, dx : int, dy :int):
    '''
    func ForwardPass()
    
    :type firstLayer: list{weights:[... : int](2), bias, result}
    :type outerLayer: list{weights:[... : int](4), bias, result}
    :type dx: int
    :type dy: int

    Specifically designed for 2-4-4

    Debug

    '''

    prediction = None

    for iNeuron in range(len(firstLayer)):
        # The result of the neuron is equal to dx*w0 + dx*w1 + b
        firstLayer[iNeuron][2] = dx*firstLayer[iNeuron][0][1] + dy*firstLayer[iNeuron][0][1] + firstLayer[1]

    outputValues = []

    for iNeuron in range(len(outerLayer)):
        wOut = []
        for jNeuron in range(len(firstLayer)):
            wOut.append(firstLayer[jNeuron][2] * outerLayer[iNeuron][0][jNeuron])
        outerLayer[iNeuron][2] = sum(wOut) + outerLayer[iNeuron][1]
        outputValues.append(sum(wOut) + outerLayer[iNeuron][1])


    if len(set(outputValues)) == 1:
        randomPrediction = random.randint(0,3)
        prediction = [randomPrediction, outputValues[randomPrediction]]
    else:
        maxValue = max(outputValues)
        maxIndex = outputValues.index(maxValue)
        prediction = [maxIndex, maxValue] #will format (0..3, val) into (dir, val)    

    return prediction



def Backpropagate(oNeuronIndex : int, outerLayer : list, firstLayer : list, TDerror : int, loss : int, Alpha : float, dx : int, dy : int):
    
    '''
    func Backpropagate()
    :type oNeuronIndex: int
    :type outerLayer: list{weights:[... : int](4), bias, result}
    :type firstLayer: list{weights:[... : int](2), bias, result}
    :type TDerror: int, error
    :type loss: int, TDerror^2
    :type Alpha: float, learning rate
    :type dx: int, offset of X
    :type dy: int, offset of Y

    Specifically designed for 2-4-4

    '''

    outerLayer[oNeuronIndex][1] += Alpha * TDerror #change bias
    outputNeuronWeights = outerLayer[oNeuronIndex][0].copy() #hidden -> output weights

    #adjust h -> o weights

    for iWeight in range(len(outputNeuronWeights)):
        #responsible neuron    corr. weight    lr      error     res of corr. neuron
        outerLayer[oNeuronIndex][0][iWeight] += Alpha * TDerror * RectifiedLinearUnit(firstLayer[iWeight][2])[0]

    #adjust in -> h weights

    for iWeight in range(len(firstLayer)):
        blameDegree = TDerror * outputNeuronWeights[iWeight]
        blameDegree = blameDegree * RectifiedLinearUnit(firstLayer[iWeight][2])[1]
        blameDegreeX = Alpha * blameDegree * dx
        blameDegreeY = Alpha * blameDegree * dy
        firstLayer[iWeight][0][0] += blameDegreeX
        firstLayer[iWeight][0][1] += blameDegreeY

        #bias change
        firstLayer[iWeight][1] = firstLayer[iWeight][1] + (Alpha * blameDegree)



    return [firstLayer, outerLayer]
