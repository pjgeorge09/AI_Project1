import pandas as pd
import matplotlib.pyplot as plt

dfA = pd.read_csv('groupA.txt', header=None, names = ['Height', 'Weight', 'Gender'])
dfB = pd.read_csv('groupB.txt', header=None, names = ['Height', 'Weight', 'Gender'])
dfC = pd.read_csv('groupC.txt', header=None, names = ['Height', 'Weight', 'Gender'])
print(dfA)


'''Splits the original bulk data into two'''
'''https://stackoverflow.com/questions/27900733/python-pandas-separate-a-dataframe-based-on-a-column-value'''
malesA = dfA[dfA.Gender == 0]
femalesA = dfA[dfA.Gender == 1]
malesB = dfB[dfB.Gender == 0]
femalesB = dfB[dfB.Gender == 1]
malesC = dfC[dfC.Gender == 0]
femalesC = dfC[dfC.Gender == 1]

'''Need to NORMALIZE data!'''
'''For normalizing maybe 
    Data is already normalized in plot?'''
maxAH = dfA['Height'].max()
maxAW = dfA['Weight'].max()
'''--------'''
maxBH = dfB['Height'].max()
maxBW = dfB['Weight'].max()
'''--------'''
maxCH = dfC['Height'].max()
maxCW = dfC['Weight'].max()
'''Normalize method below
males ['Height'] = males['Height'].divide(maxH)
'''

#femalesA['Height'].divide(maxAH)
'''How to manipulate DF values now'''


'''Plotting the Data A'''
plt.scatter(femalesA['Weight'], femalesA['Height'], color='r', marker='.')
plt.scatter(malesA['Weight'], malesA['Height'], color='b', marker='.')
plt.show()
'''Plotting the Data B'''
plt.scatter(femalesB['Weight'], femalesB['Height'], color='r', marker='.')
plt.scatter(malesB['Weight'], malesB['Height'], color='b', marker='.')
plt.show()
'''Plotting the Data C'''
plt.scatter(femalesC['Weight'], femalesC['Height'], color='r', marker='.')
plt.scatter(malesC['Weight'], malesC['Height'], color='b', marker='.')
plt.show()

print(malesC['Height'].max())
'''Need to add groupB/C.txt data'''

def normalizeData (aPanda):
    maxH = aPanda['Height'].max()
    maxW = aPanda['Weight'].max()
    aPanda['Height'] = aPanda['Height'].divide(maxH)
    aPanda['Weight'] = aPanda['Weight'].divide(maxW)
    return aPanda;
    

