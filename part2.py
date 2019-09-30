###############################################################################
### Project Name : AI Project 1                                             ###
### Class        : CMSC409 Fall 2019                                        ###
### Team         : Peter George , Daniel Webster , Joseph Longo             ###
###############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

'''A method to normalize sets of data to values [0-1]'''
def normalizeData (aPanda):
    maxH = aPanda['Height'].max()
    maxW = aPanda['Weight'].max()
    aPanda['Height'] = aPanda['Height'].divide(maxH)
    aPanda['Weight'] = aPanda['Weight'].divide(maxW)
    return aPanda;

'''A method to plot the graphs based on inputs'''
def graphIt (femaleData, maleData):
    plt.scatter(femaleData['Weight'], femaleData['Height'], color='r', marker='.')
    plt.scatter(maleData['Weight'], maleData['Height'], color='b', marker='.')
    '''Find the line here, nested, using Hebbian'''
    return;

def plotLine(A,B,C):
    
    return

def initArray(anArray):
    for i in range(0,3):
        anArray.append(random.uniform(-0.5,0.5))
        anArray[i] = round(anArray[i],6)
    return anArray

# Section 7, slide 7 Hard Activation Net Value
def netCalc(array1, array2):
    sum = 0
    for i in range (0,3):
        sum = sum + array1[i]*array2[i]
        sum = round(sum,6)
    # If this value is greater than zero , fire neuron otherwise don't
    return sum

'''originalArray = initRandom array, trainingSet = dataset using, p = amount of dataset using (ex, 0.75 = 75%'''
def HAF(originalArray, trainingSet, p):
    fire = 0 #Did the neuron fire
    r,c = trainingSet.shape
    r = r/2 #Doing one male one female each time in for loop
    r = round(r * p) #Make sure it's an int
    u = 2000 # To offset for female sampling
    # It will always do this exactly one time for a male, and one time for a female
    for x in range(0,100): #statically trying different values for data update
        fire = 0
        print(x)
        '''For 1 male'''
        
        pattern = trainingSet.iloc[x] #should be one row of the passed data set
        #this block just makes the row into an array
        pArray = []
        for j in pattern:
            pArray.append(j)
        # net is  the net from slides, where it's >= -1  (midslide pp07)    
        net = netCalc(originalArray, pArray)
        print("net = " + str(net))
        #Activation function
        if (net > 0):
            fire = 1
        elif (net <= 0 ):
            fire = 0
        #Question for TA : our D is the 3rd column, our O is the activation value?
        thing = (pArray[2] - fire) #This number can either be 1 or zero now    
        
        deltaW = thing*alpha #using a static alpha right now
        print(deltaW)
        if(deltaW != 0 ):
            print("UPDATING UPDATING UPDATING UPDATING UPDATING UPDATING UPDATING ")
            # multiply deltaW by pArray
            pArray = [deltaW * i for i in pArray]
            print("pArary is " + str(pArray))
            for s in range (0,3):
                originalArray[s] = originalArray[s] + pArray[s]
                originalArray[s] = round(originalArray[s],6)
        print(originalArray)
        print(str(originalArray[0]) + "X * " + str(originalArray[1]) + "Y + " + str(originalArray[2]) + " > 0")  
        
        '''For 1 female       NOTE BELOW THIS'''
        fire = 0
        pattern = trainingSet.iloc[x+u] #should be one row of the passed data set on FEMALE side
        pArray2 = []
        for j in pattern:
            pArray2.append(j)
        net = netCalc(originalArray, pArray2)
        print("net = " + str(net))

        #Activation function
        if (net > 0):
            fire = 1
        elif (net <= 0 ):
            fire = 0
        #Question for TA : our D is the 3rd column, our O is the activation value?
        thing = (pArray2[2] - fire) #This number can either be 1 or zero now    
        
        deltaW = thing*alpha #using a static alpha right now
        print(deltaW)
        if(deltaW != 0 ):
            print("UPDATING UPDATING UPDATING UPDATING UPDATING UPDATING UPDATING ")

            # multiply deltaW by pArray
            pArray2 = [deltaW * i for i in pArray2]
            print("pArary2 is " + str(pArray2))

            for s in range (0,3):
                originalArray[s] = originalArray[s] + pArray2[s]
                originalArray[s] = round(originalArray[s],6)
        print(originalArray)

        print(str(originalArray[0]) + "X * " + str(originalArray[1]) + "Y + " + str(originalArray[2]) + " > 0")  
        
        
    return

    
'''Create Data Objects'''
dfA = pd.read_csv('groupA.txt', header=None, names = ['Height', 'Weight', 'Gender'])
dfB = pd.read_csv('groupB.txt', header=None, names = ['Height', 'Weight', 'Gender'])
dfC = pd.read_csv('groupC.txt', header=None, names = ['Height', 'Weight', 'Gender'])

'''Normalize Data'''
dfA = normalizeData(dfA)
dfB = normalizeData(dfB)
dfC = normalizeData(dfC)


'''Splits the original bulk data into two'''
'''https://stackoverflow.com/questions/27900733/python-pandas-separate-a-dataf
rame-based-on-a-column-value'''
malesA = dfA[dfA.Gender == 0]
femalesA = dfA[dfA.Gender == 1]
malesB = dfB[dfB.Gender == 0]
femalesB = dfB[dfB.Gender == 1]
malesC = dfC[dfC.Gender == 0]
femalesC = dfC[dfC.Gender == 1]

# Alpha should be dynamic
alpha = 0.001
ni = 5000 #stopping criteria
# Gain should be dynamic 
gain = 0.5
ourArray = []
ourArray = initArray(ourArray) #quick method to initialize array with 3 values between (-0.5,0.5)

'''Hard Activation Function'''
#pass it the array of weights, we pass it the data set row, 
#Calculate the net, compare it to some value

# New Array = Do alpha times x (which is our array of weights) times (d - o) from activation function
# weights = old weights plus the newArray
HAF(ourArray, dfC, 0.75)



'''
graphIt(femalesA, malesA)
plt.show()    
graphIt(femalesB, malesB)
plt.show()    
graphIt(femalesC, malesC)
plt.show()    
'''

    

