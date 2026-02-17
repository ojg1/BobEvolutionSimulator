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


def ForwardPass(firstLayer : list, outerLayer : list, dx : int, dy :int):
    '''
    func ForwardPass()
    
    :type firstLayer: list{weights:[... : int](2), bias, result}
    :type outerLayer: list{weights:[... : int](4), bias, result}
    :type dx: int
    :type dy: int

    Specifically designed for 2-4-4

    '''



def Backpropagate(oNeuronIndex : int, outerLayer : list, firstLayer : list, TDerror:int, loss:int, Alpha : int, xinp : int, yinp: int):
    
    '''
    func Backpropagate()
    :type oNeuronIndex: int
    :type outerLayer: list{weights:[... : int](4), bias, result}
    :type firstLayer: list{weights:[... : int](2), bias, result}
    :type TDerror: int, error
    :type loss: int, TDerror^2
    :type Alpha: int, learning rate

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
        blameDegreeX = Alpha * blameDegree * xinp
        blameDegreeY = Alpha * blameDegree * yinp
        firstLayer[iWeight][0][0] += blameDegreeX
        firstLayer[iWeight][0][1] += blameDegreeY

        #bias change
        firstLayer[iWeight][1] = firstLayer[iWeight][1] + (Alpha * blameDegree)



    return [firstLayer, outerLayer]



    


