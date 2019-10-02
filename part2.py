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
    aPanda['Height'] = (aPanda['Height'] - minH) / (maxH - minH)
    aPanda['Weight'] = (aPanda['Weight'] - minW) / (maxW - minW)
    return aPanda;

'''A method to plot the graphs based on inputs'''
def graphIt (femaleData, maleData):
    plt.scatter(femaleData['Weight'], femaleData['Height'], color='r', marker='.')
    plt.scatter(maleData['Weight'], maleData['Height'], color='b', marker='.')
    '''Find the line here, nested, using Hebbian'''
    return;

def initArray(anArray):
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


def SAF(originalArray, trainingSet, p, epsilon, alpha, gain):
    ni = 5000
    fire = 0 #initialize for reset/global use in function, "did neuron fire"
    r,c = trainingSet.shape
    #r = r/2 #Doing one male one female each time in for loop
    #r = round(r * p) #Make sure it's an int
    u = 2000 # To offset for female sampling
    # It will always do this exactly one time for a male, and one time for a female
    TE = 2000
    while(TE > epsilon):
        for bigNum in range (0,500):
            TE = 0
            print(bigNum)
            #print(str(originalArray[0]) + "X * " + str(originalArray[1]) + "Y + " + str(originalArray[2]) + " > 0")  
            for x in range(0,1999): #statically trying different values for data update
                fire = 0
                '''For 1 male'''
                #ex = random.randrange(0,1999)
                pattern = trainingSet.iloc[x] #should be one row of the passed data set
                #this block just makes the row into an array
                pArray = []
                for j in pattern:
                    pArray.append(j)
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
                pattern2 = trainingSet.iloc[x+u] #should be one row of the passed data set on FEMALE side
                pArray2 = []
                for j in pattern2:
                    pArray2.append(j)
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
            print(TE)
            print(str(TE) + " < " + str(epsilon))   
            if(TE<epsilon or bigNum == 99):
                return originalArray
    return originalArray



'''                 Hard Activation Function Method                         '''
''' @input originalArray = is the starting weight array, initialized elsewhere
    @input trainingSet = is the set of data, dfA,dfB,dfC for us in our work
    @input p = is the percent of the data you want to use as training data'''
def HAF(originalArray, trainingSet, p, epsilon, alpha):
    ni = 5000
    fire = 0 #initialize for reset/global use in function, "did neuron fire"
    r,c = trainingSet.shape
    #r = r/2 #Doing one male one female each time in for loop
    #r = round(r * p) #Make sure it's an int
    u = 2000 # To offset for female sampling
    # It will always do this exactly one time for a male, and one time for a female
    TE = 2000
    while(TE > epsilon):
        TE = 0
        for bigNum in range (0,ni):
            TE = 0
            print(bigNum)
            #print(str(originalArray[0]) + "X * " + str(originalArray[1]) + "Y + " + str(originalArray[2]) + " > 0")  
            for x in range(0,1999): #statically trying different values for data update
                fire = 0
                '''For 1 male'''
                #ex = random.randrange(0,1999)
                pattern = trainingSet.iloc[x] #should be one row of the passed data set
                #this block just makes the row into an array
                pArray = []
                for j in pattern:
                    pArray.append(j)
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
                pattern2 = trainingSet.iloc[x+u] #should be one row of the passed data set on FEMALE side
                pArray2 = []
                for j in pattern2:
                    pArray2.append(j)
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
            print(TE)
            if(TE<epsilon or bigNum == ni):
                   return originalArray
    return originalArray

'''Can be modified to plot only the non-training data just add a p parameter 1-p'''
def plotIt(theDF, theEQ):
    males = theDF[theDF.Gender == 0]
    females = theDF[theDF.Gender == 1]
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
dfA = pd.read_csv('groupA.txt', header=None, names = ['Height', 'Weight', 'Gender'])
dfB = pd.read_csv('groupB.txt', header=None, names = ['Height', 'Weight', 'Gender'])
dfC = pd.read_csv('groupC.txt', header=None, names = ['Height', 'Weight', 'Gender'])

'''Normalize Data'''
dfA = normalizeData(dfA)
dfB = normalizeData(dfB)
dfC = normalizeData(dfC)

'''Define Epsilons'''
epsilonA = 0.00001
epsilonB = 100
epsilonC = 1450


# Alpha should be dynamic
alphaAB = 0.8
ni = 5000 #stopping criteria
# Gain should be dynamic 
gain2 = 50
ourArray = []

ourArray = initArray(ourArray) #quick method to initialize array with 3 values between (-0.5,0.5)

'''Hard Activation Function'''

'''weightsHAFA = HAF(ourArray, dfA, 1, epsilonA, 0.5)
plotIt(dfA,weightsHAFA)
weightsHAFB = HAF(ourArray, dfB, 1, epsilonB, 0.5)
plotIt(dfB,weightsHAFB)
weightsHAFC = HAF(ourArray, dfC, 1, epsilonC, 0.0001)
plotIt(dfC,weightsHAFC)'''

#SAF(ourArray,dfA,0.75)
weightsSAFC = SAF(initArray(ourArray), dfC, 1, 0.5, epsilonC, 0.8)
plotIt(dfC, weightsSAFC)

weightsSAFB = SAF(ourArray, dfB, 1, 1, epsilonB, 0.8)
plotIt(dfB, weightsSAFB)
weightsSAFA = SAF(ourArray, dfA, 1, 1, epsilonA, 0.8)
plotIt(dfA, weightsSAFA)


    

