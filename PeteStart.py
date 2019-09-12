import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''A method to normalize sets of data to values [0-1]'''
def normalizeData (aPanda):
    maxH = aPanda['Height'].max()
    maxW = aPanda['Weight'].max()
    aPanda['Height'] = aPanda['Height'].divide(maxH)
    aPanda['Weight'] = aPanda['Weight'].divide(maxW)
    return aPanda;

'''A method to be used in graphIt to generate a best-fit line by Hebbian learning'''
def hebb (data, data2, plot):
    bias = 1
    
    return;

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


    

