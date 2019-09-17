###############################################################################
### Project Name : AI Project 1                                             ###
### Class        : CMSC409 Fall 2019                                        ###
### Team         : Peter George , Daniel Webster , Joseph Longo             ###
###############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
import seaborn as sn

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

'''Create Data Objects'''
dfA = pd.read_csv('groupA.txt', header=None, names = ['Height', 'Weight', 'Gender'])
dfB = pd.read_csv('groupB.txt', header=None, names = ['Height', 'Weight', 'Gender'])
dfC = pd.read_csv('groupC.txt', header=None, names = ['Height', 'Weight', 'Gender'])
print(dfA)


'''Normalize Data'''
dfA = normalizeData(dfA)
dfB = normalizeData(dfB)
dfC = normalizeData(dfC)


'''Splits the original bulk data into two'''
'''https://stackoverflow.com/questions/27900733/python-pandas-separate-a-dataframe-based-on-a-column-value'''
malesA = dfA[dfA.Gender == 0]
femalesA = dfA[dfA.Gender == 1]
malesB = dfB[dfB.Gender == 0]
femalesB = dfB[dfB.Gender == 1]
malesC = dfC[dfC.Gender == 0]
femalesC = dfC[dfC.Gender == 1]


'''Plotting the Data A'''
graphIt(femalesA, malesA)
'''Self created line for A'''
x = np.linspace(0.75,1,100)
y = (-0.96)*x+1.7
plt.plot(x, y,'-g')
plt.show()    
'''Plotting the Data B'''
graphIt(femalesB, malesB)
'''Self created line for A'''
x = np.linspace(0.75,1,100)
y = (-1)*x+1.825
plt.plot(x, y,'-g')
plt.show()    
'''Plotting the Data C'''
graphIt(femalesC, malesC)
'''Self created line for A'''
x = np.linspace(0.85,1,100)
y = (-1.2)*x+2
plt.plot(x, y,'-g')
plt.show()    

'''Calculate Confusion Matrix-must install pandas_ml pakage'''
groupA=pd.DataFrame(dfA)
groupB=pd.DataFrame(dfB)
groupC=pd.DataFrame(dfC)

def calculate_inequalA(num):
   if (num['Height']+0.96*num['Weight']-1.7)>0:
       return 0
   else:
       return 1

def calculate_inequalB(num):
   if (num['Height']+1*num['Weight']-1.825)>0:
       return 0
   else:
       return 1

def calculate_inequalC(num):
   if (num['Height']+1.2*num['Weight']-2)>0:
       return 1
   else:
       return 0
   
    
print(groupA.apply(calculate_inequalA,axis=1))

print(dfA['Gender'])

predicted_matrixA=groupA.apply(calculate_inequalA,axis=1)

actual_matrixA=dfA['Gender']

predicted_matrixB=groupB.apply(calculate_inequalA,axis=1)

actual_matrixB=dfB['Gender']

predicted_matrixC=groupC.apply(calculate_inequalA,axis=1)

actual_matrixC=dfC['Gender']


Confusion_MatrixA = ConfusionMatrix( actual_matrixA, predicted_matrixA)
Confusion_MatrixA.print_stats()
print('\n\n')
confusion_matrixA = pd.crosstab(actual_matrixA, predicted_matrixA)
print (confusion_matrixA)
print('\n\n')
Confusion_MatrixB = ConfusionMatrix(actual_matrixB, predicted_matrixB)
Confusion_MatrixB.print_stats()
print('\n\n')
confusion_matrixB = pd.crosstab(actual_matrixB, predicted_matrixB)
print (confusion_matrixB)
print('\n\n')
Confusion_MatrixC = ConfusionMatrix(actual_matrixC, predicted_matrixC)
Confusion_MatrixC.print_stats()
print('\n\n')
confusion_matrixC = pd.crosstab(actual_matrixC, predicted_matrixC)
print (confusion_matrixC)
print('\n\n')    

