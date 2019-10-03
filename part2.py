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
    plt.scatter(femaleData['Weight'], femaleData['Height'], color='r', marker='v', s=7, label="Female", alpha=0.5)
    plt.scatter(maleData['Weight'], maleData['Height'], color='b', marker='.', s=7, label="Male", alpha=0.5)
    plt.legend()
    '''Find the line here, nested, using Hebbian'''
    return;

'''Can be modified to plot only the non-training data just add a p parameter 1-p'''
def plotIt(theDF, theDFo, theEQ, graphTitle):
    males = theDFo[theDFo.Gender == 0]
    females = theDFo[theDFo.Gender == 1]
    # Plotting the line        
    x = np.linspace(0,1,100)
    b = -theEQ[2]/theEQ[1]
    slope = -theEQ[0]/theEQ[1]
    y = slope*x+b
    graphIt(females, males)
    plt.plot(x,y,'-g')
    plt.axis([0,1.1,0,1.1])
    plt.xlabel('Heights, normalized')
    plt.ylabel('Weights, normalized')
    plt.title(graphTitle, loc='center')
    plt.show()
    return

'''A method to initialize arrays with 3 values within the range below, returns that random array'''
def initArray():
    anArray = []
    for i in range(0,3):
        anArray.append(random.uniform(-0.5,0.5))
    return anArray

'''A method that calculates the net where net is (Σ(W_i)(X_i)) + bias'''
def netCalc(array1, array2):
    sum = 0
    for i in range (0,2):                           #Dot Product & round
        sum = sum + array1[i]*array2[i]
    sum += array1[2]                         #Adding in the bias
    # If this value is greater than zero , fire neuron otherwise don't (for HAF)
    return sum

'''Soft Activation Function'''
'''A method to calculate a best fit line based on criteria and updates, with no hard definition (nonbinary)
   @input trainingSet : the set of data we are learning on
   @p : the percent of the data we want to train on in range (0,1) should not be zero
   @epsilon : A stopping criteria that determines if line is good enough compared to total error
   @alpha : learning rate, set to be static and defined elsewhere, independent for each trial
   @gain : the gaining rate to be used in the equation
   RETURN : returns an array in the form of [A B C] for the formula Ax + By + C < 0 '''
def SAF(trainingSet, p, epsilon, alpha, gain):
    originalArray = initArray()
    r = len(trainingSet)
    r = r/2 #Doing one male one female each time in for loop
    r = round(r * p) #Make sure it's an int
    offset = 2000 # To offset for female sampling
    # It will always do this exactly one time for a male, and one time for a female
    TE = 2000
    while(TE > epsilon):
        for bigNum in range (0,ni):
            TE = 0
            for x in range(0,r): #statically trying different values for data update
                '''For 1 male'''
                pattern = trainingSet[x]
                #this block just makes the row into an array
                pArray = []
                for j in pattern:
                    pArray.append(pattern[j])
                # net is  the net from slides, where it's >= -1  (midslide pp07)
                net = netCalc(originalArray,pArray)
                out = 1/(1+(math.exp(-1*gain*net)))
                delta = pArray[2] - out 
                for s in range (0,2):
                    originalArray[s] += delta*alpha*pArray[s]
                originalArray[2] += delta*alpha
                TE += delta**2
            
                '''For 1 female       NOTE BELOW THIS'''
                pattern2 = trainingSet[x+offset]
                pArray2 = []
                for j in pattern2:
                    pArray2.append(pattern2[j])
                    
                net2 = netCalc(originalArray,pArray2)
                out = 1/(1+(math.exp(-gain*net2)))
                
                delta = pArray2[2] - out
                for s in range (0,2):
                    originalArray[s] += delta*alpha*pArray2[s]
                originalArray[2] += delta*alpha            
                TE += delta**2
            if(TE<epsilon):
                return originalArray
    return originalArray



'''                 Hard Activation Function Method                         '''
''' A method to calculate a best fit line based on criteria and updates, with a hard definition (binary)
    @input trainingSet : is the set of data, dfA,dfB,dfC for us in our work
    @input p : is the percent of the data you want to use as training data
    @epsilon : A stopping criteria that determines if line is good enough compared to total error
    @alpha : learning rate, set to be static and defined elsewhere, independent for each trial '''
def HAF(trainingSet, p, epsilon, alpha):
    originalArray = initArray()
    fire = 0 #initialize for reset/global use in function, "did neuron fire"
    r = len(trainingSet)
    r = r/2 #Doing one male one female each time in for loop
    r = round(r * p) #Make sure it's an int
    offset = 2000 # To offset for female sampling
    # It will always do this exactly one time for a male, and one time for a female
    TE = 2000
    while(TE > epsilon):
        TE = 0
        for bigNum in range (0,ni):
            TE = 0
            for x in range(0,r): #statically trying different values for data update
                fire = 0
                '''For 1 male'''
                pattern = trainingSet[x]
                #this block just makes the row into an array
                pArray = []
                for j in pattern:
                    pArray.append(pattern[j])
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
                pattern2 = trainingSet[x+offset]
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
            if(TE<epsilon):
                   return originalArray
    return originalArray

def partial(aDF, aPercent):
    combinedDF = pd.DataFrame(columns=['Height', 'Weight', 'Gender'])
    r,c = aDF.shape
    length = r
    for _ in range(0, round((length*aPercent)/2)):
        combinedDF = combinedDF.append(aDF.iloc[_])
    for _ in range(2000, 2000+round((length*aPercent)/2)):
        combinedDF = combinedDF.append(aDF.iloc[_])
    
    
    return combinedDF

'''Create Data Objects'''
dfAn = pd.read_csv('groupA.txt', header=None, names = ['Height', 'Weight', 'Gender'])
dfBn = pd.read_csv('groupB.txt', header=None, names = ['Height', 'Weight', 'Gender'])
dfCn = pd.read_csv('groupC.txt', header=None, names = ['Height', 'Weight', 'Gender'])

'''Normalize Data'''
dfAo = normalizeData(dfAn)
dfBo = normalizeData(dfBn)
dfCo = normalizeData(dfCn)
'''Create training sets'''
dfA75 = partial(dfAo, 0.75)
dfB75 = partial(dfBo, 0.75)
dfC75 = partial(dfCo, 0.75)
dfA25 = partial(dfAo, 0.25)
dfB25 = partial(dfBo, 0.25)
dfC25 = partial(dfCo, 0.25)

'''Turn DF objects to lists for speed'''
dfAn = dfAn.to_dict('index')
dfBn = dfBn.to_dict('index')
dfCn = dfCn.to_dict('index')

dfA = dfAo.to_dict('index')
dfB = dfBo.to_dict('index')
dfC = dfCo.to_dict('index')


'''Define Variables'''
ε_A = 0.00001
ε_B = 100
ε_C = 1450
ni = 5000

'''Hard Activation Function at 75% training'''
ht75 = 0.75
weightsHAFA = HAF(dfA, ht75, ε_A, 0.5)
plotIt(dfA, dfA75, weightsHAFA, "HAF Train on 75% of A")
weightsHAFB = HAF(dfB, ht75, ε_B, 0.5)
plotIt(dfB, dfB75, weightsHAFB, "HAF Train on 75% of B")
weightsHAFC = HAF(dfC, ht75, ε_C, 0.01)
plotIt(dfC, dfC75, weightsHAFC, "HAF Train on 75% of C")

'''Soft Activation Function at 75% training'''
st75 = 0.75
weightsSAFA = SAF(dfA, st75, ε_A, 0.5, 5)
plotIt(dfA, dfA75, weightsSAFA, "SAF Train on 75% of A")
weightsSAFB = SAF(dfB, st75, ε_B, 0.5, 5)
plotIt(dfB, dfB75, weightsSAFB, "SAF Train on 75% of B")
weightsSAFC = SAF(dfC, st75, ε_C, 0.5, 5)
plotIt(dfC, dfC75, weightsSAFC, "SAF Train on 75% of C")

'''Hard Activation Function at 25% training'''
ht25 = 0.25
weightsHAFA = HAF(dfA, ht25, ε_A, 0.5)
plotIt(dfA, dfA25, weightsHAFA, "HAF Train on 25% of A")
weightsHAFB = HAF(dfB, ht25, ε_B, 0.5)
plotIt(dfB, dfB25, weightsHAFB, "HAF Train on 25% of B")
weightsHAFC = HAF(dfC, ht25, ε_C, 0.5)
plotIt(dfC, dfC25, weightsHAFC, "HAF Train on 25% of C")

'''Soft Activation Function at 25% training'''
st25 = 0.25
weightsSAFA = SAF(dfA, st25, ε_A, 0.5, 5)
plotIt(dfA, dfA25, weightsSAFA, "SAF Train on 25% of A")
weightsSAFB = SAF(dfB, st25, ε_B, 0.5, 5)
plotIt(dfB, dfB25, weightsSAFB, "SAF Train on 25% of B")
weightsSAFC = SAF(dfC, st25, ε_C, 0.5, 5)
plotIt(dfC, dfC25, weightsSAFC, "SAF Train on 25% of C")

def calculate_inequal(num,ww):
   if (num['Height']*ww[0]+ww[1]*num['Weight']+ww[2])>0:
       return 1
   else:
       return 0
'''Testing Function
    @input:dataframe,weights from weightsSAF or HAF
    @output:error for testing with actual and plots for testing'''

def testingFunction(df_train,weights,percnt, stringTitle):
    r=4000
    r=r/2
    r=round(r*percnt)
    nw=round(len(df_train)/2)
    rlist=[]
    for i in range(r,nw):
        rlist.append(df_train[i])
    
    for l in range(nw+r,len(df_train)):
        rlist.append(df_train[l])
    
    
    df_test=pd.DataFrame(rlist)
#    print(df_test)
    df_testPerceptrn=df_test.apply(calculate_inequal,axis=1,ww=(weights))
#    print(df_testPerceptrn)
    actual_matrix=df_test['Gender']
    
    predicted_matrix=df_testPerceptrn
    
    confusion_matrix=pd.crosstab(actual_matrix,predicted_matrix)
    
    df = normalizeData(df_test)
    plotIt(df_test,df,weights,stringTitle)
    
    print(confusion_matrix)

    return

#print(males)
#print(females)
#testingFunction(dfA,weightsSAFA,0.75)


'''Hard Activation Function at 25% testing'''
print("\nHard Activation Function at 25% testing:\n")
htst75=0.75
testingFunction(dfA,weightsHAFA,htst75, "HAF Test on 25% of A")
testingFunction(dfB,weightsHAFB,htst75, "HAF Test on 25% of B")
testingFunction(dfC,weightsHAFC,htst75, "HAF Test on 25% of C")
        
'''Soft Activation Function at 25% testing'''
print("\nSoft Activation Function at 25% testing:\n")
stst75=0.75
testingFunction(dfA,weightsSAFA,stst75, "SAF Test on 25% of A")
testingFunction(dfB,weightsSAFB,stst75, "SAF Test on 25% of B")
testingFunction(dfC,weightsSAFC,stst75, "SAF Test on 25% of C")

'''Hard Activation Function at 75% testing'''
print("\nHard Activation Function at 75% testing:\n")
htst25=0.25
testingFunction(dfA,weightsHAFA,htst25, "HAF Test on 75% of A")
testingFunction(dfB,weightsHAFB,htst25, "HAF Test on 75% of B")
testingFunction(dfC,weightsHAFC,htst25, "HAF Test on 75% of C")

'''Soft Activation Function at 75% testing'''
print("\n\nSoft Activation Function at 75% testing:\n")
stst25=0.25
testingFunction(dfA,weightsSAFA,stst25, "SAF Test on 75% of A")
testingFunction(dfB,weightsSAFB,stst25, "SAF Test on 75% of B")
testingFunction(dfC,weightsSAFC,stst25, "SAF Test on 75% of C")
