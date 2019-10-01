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
    
    sum = round(sum,6)
    return sum

def SAF(orig, train, p):
    u=2000
    cycles = 100
    r,c = train.shape                               #r=rows c=columns
    r = r/2                                         #half because we do 1M1W at a time
    r = round(r * p)                                #Make sure it's an int
    alpha = 0.1                                     #Static Alpha
    TE = 0
    
    for i in range (0,cycles):                      #For number of cycles
        print(i)
        print(orig)
        for k in range (0,r):                       #For each row we are doing
            '''One Male'''
            row = train.iloc[k]                     #Make it a row object
            pArray=[]
            for l in row:
                pArray.append(l)                    #Make row object a simple array
            net = 0                                 #Init net
            for j in range (0,3):                   #For every value of the array
                net = net + pArray[j]*orig[j]       #Net += (A * B) per notes slide 13 pp11/12
            out = 1                                 #init out
            if (net < 0):
                out = -1                            #Change out if net < 0 (maybe out = 0 because unipolar)
            error = pArray[2] - out                  #Error = desired output minus actual output (0 or 1)
            learn = alpha*error                     #Learn = alpha * error per same slide
            for j in range(0,3):                    
                orig[j] = orig[j] + learn*pArray[j] #Update Weight by adding learn * pattern
            
            '''One Female'''
            row = train.iloc[k+u]                   #Make it a row object
            pArray2=[]
            for l in row:
                pArray2.append(l)                    #Make row object a simple array
            net = 0                                 #Init net
            for j in range (0,3):                   #For every value of the array
                net = net + pArray2[j]*orig[j]       #Net += (A * B) per notes slide 13 pp11/12
            out = 1                                 #init out
            if (net < 0):
                out = -1                            #Change out if net < 0 (maybe out = 0 because unipolar)
            error = pArray2[2] - out                  #Error = desired output minus actual output (0 or 1)
            learn = alpha*error                     #Learn = alpha * error per same slide
            for j in range(0,3):                    
                orig[j] = orig[j] + learn*pArray2[j] #Update Weight by adding learn * pattern
            
            
            
    return
    


'''                 Hard Activation Function Method                         '''
''' @input originalArray = is the starting weight array, initialized elsewhere
    @input trainingSet = is the set of data, dfA,dfB,dfC for us in our work
    @input p = is the percent of the data you want to use as training data'''
def HAF(originalArray, trainingSet, p):
    fire = 0 #initialize for reset/global use in function, "did neuron fire"
    r,c = trainingSet.shape
    r = r/2 #Doing one male one female each time in for loop
    r = round(r * p) #Make sure it's an int
    u = 2000 # To offset for female sampling
    # It will always do this exactly one time for a male, and one time for a female
    for bigNum in range (0,100):
        #print(bigNum)
        #print(str(originalArray[0]) + "X * " + str(originalArray[1]) + "Y + " + str(originalArray[2]) + " > 0")  
        for x in range(0,r): #statically trying different values for data update
            fire = 0
            '''For 1 male'''
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
    
    # Plotting the line        
    x = np.linspace(0,1,100)
    b = -originalArray[2]/originalArray[1]
    slope = -originalArray[0]/originalArray[1]
    y = slope*x+b
    graphIt(femalesA, malesA)
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


'''Splits the original bulk data into two'''
malesA = dfA[dfA.Gender == 0]
femalesA = dfA[dfA.Gender == 1]
malesB = dfB[dfB.Gender == 0]
femalesB = dfB[dfB.Gender == 1]
malesC = dfC[dfC.Gender == 0]
femalesC = dfC[dfC.Gender == 1]

# Alpha should be dynamic
alpha = 0.8
ni = 5000 #stopping criteria
# Gain should be dynamic 
gain = 10
ourArray = []
ourArray = initArray(ourArray) #quick method to initialize array with 3 values between (-0.5,0.5)

'''Hard Activation Function'''
#pass it the array of weights, we pass it the data set row, 
#Calculate the net, compare it to some value

# New Array = Do alpha times x (which is our array of weights) times (d - o) from activation function
# weights = old weights plus the newArray
weightsHAFA = HAF(ourArray, dfA, 0.75)
weightsHAFA = HAF(ourArray, dfB, 0.75)
weightsHAFA = HAF(ourArray, dfC, 0.75)

#SAF(ourArray,dfA,0.75)


'''
graphIt(femalesA, malesA)
plt.show()    
graphIt(femalesB, malesB)
plt.show()    
graphIt(femalesC, malesC)
plt.show()    
'''

    

