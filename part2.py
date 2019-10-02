###############################################################################
### Project Name : AI Project 1                                             ###
### Class        : CMSC409 Fall 2019                                        ###
### Team         : Peter George , Daniel Webster , Joseph Longo             ###
###############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

'''A method to normalize sets of data to values [0-1]'''
def normalizeData (aPanda):
    maxH = aPanda['Height'].max()
    maxW = aPanda['Weight'].max()
    minH = aPanda['Height'].min()
    minW = aPanda['Weight'].min()
    aPanda['Height'] = round((aPanda['Height'] - minH) / (maxH - minH),6)
    aPanda['Weight'] = round((aPanda['Weight'] - minW) / (maxW - minW),6)
    return aPanda;

'''A method to plot the graphs based on inputs'''
def graphIt (femaleData, maleData):
    plt.scatter(femaleData['Weight'], femaleData['Height'], color='r', marker='.')
    plt.scatter(maleData['Weight'], maleData['Height'], color='b', marker='.')
    '''Find the line here, nested, using Hebbian'''
    return;

def initArray():
    anArray = []
    for i in range(0,3):
        anArray.append(random.uniform(-0.5,0.5))
        anArray[i] = round(anArray[i],6)
    return anArray

# Section 7, slide 7 Hard Activation Net Value
def netCalc(array1, array2):
    sum = 0
    for i in range (0,2):                           #Dot Product & round
        sum = sum + array1[i]*array2[i]
    sum += array1[2]                         #Adding in the bias
    # If this value is greater than zero , fire neuron otherwise don't
    
    return sum


def SAF(trainingSet, p, epsilon, alpha, gain):
    originalArray = initArray()
    ni = 5000
    fire = 0 #initialize for reset/global use in function, "did neuron fire"
    #r,c = trainingSet.shape
    r=4000
    r = r/2 #Doing one male one female each time in for loop
    r = round(r * p) #Make sure it's an int
    u = 2000 # To offset for female sampling
    # It will always do this exactly one time for a male, and one time for a female
    TE = 2000
    while(TE > epsilon):
        for bigNum in range (0,ni):
            TE = 0
            #print(str(originalArray[0]) + "X * " + str(originalArray[1]) + "Y + " + str(originalArray[2]) + " > 0")  
            for x in range(0,r): #statically trying different values for data update
                fire = 0
                '''For 1 male'''
                #ex = random.randrange(0,1999)
                #pattern = trainingSet.iloc[x] #should be one row of the passed data set
                pattern = trainingSet[x]
                #this block just makes the row into an array
                pArray = []
                for j in pattern:
                    pArray.append(pattern[j])
                # net is  the net from slides, where it's >= -1  (midslide pp07)
                net = netCalc(originalArray,pArray)
                try:
                    out = 1/(1+(math.exp(-1*gain*net)))
                except OverflowError:
                    out = pArray[2]
                fire = out
                delta = pArray[2] - fire 
                for s in range (0,2):
                    originalArray[s] += delta*alpha*pArray[s]
                originalArray[2] += delta*alpha
                TE += delta**2
            
                '''For 1 female       NOTE BELOW THIS'''
                fire = 0
                #pattern2 = trainingSet.iloc[x+u] #should be one row of the passed data set on FEMALE side
                pattern2 = trainingSet[x+u]
                pArray2 = []
                for j in pattern2:
                    pArray2.append(pattern2[j])
                net2 = netCalc(originalArray,pArray2)
                try:
                    out = 1/(1+(math.exp(-gain*net2)))
                except OverflowError:
                    out = pArray2[2]
                fire = out
                delta = pArray2[2] - fire
                for s in range (0,2):
                    originalArray[s] += delta*alpha*pArray2[s]
                originalArray[2] += delta*alpha            
                TE += delta**2
            TE = round(TE,6)
            if(TE<epsilon or bigNum == 99):
                return originalArray
    return originalArray



'''                 Hard Activation Function Method                         '''
''' @input originalArray = is the starting weight array, initialized elsewhere
    @input trainingSet = is the set of data, dfA,dfB,dfC for us in our work
    @input p = is the percent of the data you want to use as training data'''
def HAF(trainingSet, p, epsilon, alpha):
    originalArray = initArray()
    ni = 5000
    fire = 0 #initialize for reset/global use in function, "did neuron fire"
    r = 4000
    #r,c = trainingSet.shape
    r = r/2 #Doing one male one female each time in for loop
    r = round(r * p) #Make sure it's an int
    u = 2000 # To offset for female sampling
    # It will always do this exactly one time for a male, and one time for a female
    TE = 2000
    while(TE > epsilon):
        TE = 0
        for bigNum in range (0,ni):
            TE = 0
            #print(str(originalArray[0]) + "X * " + str(originalArray[1]) + "Y + " + str(originalArray[2]) + " > 0")  
            for x in range(0,r): #statically trying different values for data update
                fire = 0
                '''For 1 male'''
                #ex = random.randrange(0,1999)
                #pattern = trainingSet.iloc[x] #should be one row of the passed data set
                pattern = trainingSet[x]
                #this block just makes the row into an array
                pArray = []
                for j in pattern:
                    pArray.append(pattern[j])
                # net is  the net from slides, where it's >= -1  (midslide pp07)
                net = netCalc(originalArray, pArray) #USE PP4 SLIDE 3,
                #Activation function
                if (net > 0):
                    fire = 1
                elif (net <= 0 ):
                    fire = 0
                delta = pArray[2] - fire 
                for s in range (0,2):
                    originalArray[s] += delta*alpha*pArray[s]
                originalArray[2] += delta*alpha
                TE += abs(delta)
            
                '''For 1 female       NOTE BELOW THIS'''
                fire = 0
                #pattern2 = trainingSet.iloc[x+u] #should be one row of the passed data set on FEMALE side
                pattern2 = trainingSet[x+u]
                pArray2 = []
                for j in pattern2:
                    pArray2.append(pattern2[j])
                net2 = netCalc(originalArray, pArray2)
                #Activation function
                if (net2 > 0):
                    fire = 1
                elif (net2 <= 0 ):
                    fire = 0             
                delta = pArray2[2] - fire
                for s in range (0,2):
                    originalArray[s] += delta*alpha*pArray2[s]
                originalArray[2] += delta*alpha            
                TE += abs(delta)
            if(TE<epsilon or bigNum == ni):
                   return originalArray
    return originalArray

'''Can be modified to plot only the non-training data just add a p parameter 1-p'''
def plotIt(theDF, theDFo, theEQ):
    males = theDFo[theDFo.Gender == 0]
    females = theDFo[theDFo.Gender == 1]
    # Plotting the line        
    x = np.linspace(0,1,100)
    b = -theEQ[2]/theEQ[1]
    slope = -theEQ[0]/theEQ[1]
    y = slope*x+b
    graphIt(females, males)
    plt.plot(x,y,'-g')
    plt.show()
    return
    
'''Create Data Objects'''
dfAo = pd.read_csv('groupA.txt', header=None, names = ['Height', 'Weight', 'Gender'])
dfBo = pd.read_csv('groupB.txt', header=None, names = ['Height', 'Weight', 'Gender'])
dfCo = pd.read_csv('groupC.txt', header=None, names = ['Height', 'Weight', 'Gender'])
'''Normalize Data'''
dfAo = normalizeData(dfAo)
dfBo = normalizeData(dfBo)
dfCo = normalizeData(dfCo)
dfA = dfAo.to_dict('index')
dfB = dfBo.to_dict('index')
dfC = dfCo.to_dict('index')

print(dfA[0])
thing = dfA[0]
for _ in thing:
    print(thing[_])
'''Define Epsilons'''
epsilonA = 0.00001
epsilonB = 100
epsilonC = 1450

'''Hard Activation Function at 75% training'''
ht75 = 0.75
weightsHAFA = HAF(dfA, ht75, epsilonA, 0.5)
plotIt(dfA, dfAo, weightsHAFA)
weightsHAFB = HAF(dfB, ht75, epsilonB, 0.5)
plotIt(dfB, dfBo, weightsHAFB)
weightsHAFC = HAF(dfC, ht75, epsilonC, 0.01)
plotIt(dfC, dfCo, weightsHAFC)

'''Soft Activation Function at 75% training'''
st75 = 0.75
weightsSAFA = SAF(dfA, st75, epsilonA, 0.5, 5)
plotIt(dfA, dfAo, weightsSAFA)
weightsSAFB = SAF(dfB, st75, epsilonB, 0.5, 5)
plotIt(dfB, dfBo, weightsSAFB)
weightsSAFC = SAF(dfC, st75, epsilonC, 0.5, 5)
plotIt(dfC, dfCo, weightsSAFC)

'''Hard Activation Function at 25% training'''
ht25 = 0.25
weightsHAFA = HAF(dfA, ht25, epsilonA, 0.5)
plotIt(dfA, dfAo, weightsHAFA)
weightsHAFB = HAF(dfB, ht25, epsilonB, 0.5)
plotIt(dfB, dfBo, weightsHAFB)
weightsHAFC = HAF(dfC, ht25, epsilonC, 0.5)
plotIt(dfC, dfCo, weightsHAFC)

'''Soft Activation Function at 25% training'''
st25 = 0.25
weightsSAFA = SAF(dfA, st25, epsilonA, 0.5, 5)
plotIt(dfA, dfAo, weightsSAFA)
weightsSAFB = SAF(dfB, st25, epsilonB, 0.5, 5)
plotIt(dfB, dfBo, weightsSAFB)
weightsSAFC = SAF(dfC, st25, epsilonC, 0.5, 5)
plotIt(dfC, dfCo, weightsSAFC)

