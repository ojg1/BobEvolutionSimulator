import math

def DistanceMagnitude(Target, Bob, ScreenWidth, ScreenHeight):
    dx = (Target.x-Bob.x)/ScreenWidth
    dy = (Target.y-Bob.y)/ScreenHeight
    return (dx, dy)    

def RectifiedLinearUnit(z):
    

    d = 0
    if z > 0:
        d = 1
    return (max(0, z), d)

def Backpropagate(oNeuronIndex : int, outerLayer : list, firstLayer : list, loss : int, TDerror : int):
    
    '''
    func Backpropagate()
    :type oNeuronIndex: int
    :type outerLayer: list{weights:[... : int](4), bias, result}
    :type firstLayer: list{weights:[... : int](2), bias, result}
    :type loss: int
    :type TDerror: int
    
    Specifically designed for 2-4-4

    

    '''

    outputNeuron = outerLayer[oNeuronIndex]
    outputNeuronBias    = outputNeuron[2] 
    outputNeuronWeights = outputNeuron[0] #hidden
    


