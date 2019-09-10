import pandas as pd
import matplotlib.pyplot as plt

dfA = pd.read_csv('groupA.txt', header=None, names = ['Height', 'Weight', 'Gender'])
dfB = pd.read_csv('groupB.txt', header=None, names = ['Height', 'Weight', 'Gender'])
dfC = pd.read_csv('groupC.txt', header=None, names = ['Height', 'Weight', 'Gender'])

'''Appending the data groups from files'''
dfA = dfA.append(dfB, ignore_index=True)
dfA = dfA.append(dfC, ignore_index=True)

print(dfA)


'''Splits the original bulk data into two'''
'''https://stackoverflow.com/questions/27900733/python-pandas-separate-a-dataframe-based-on-a-column-value'''
males = dfA[dfA.Gender == 0]
females = dfA[dfA.Gender == 1]
print(females)



'''Need to NORMALIZE data!'''
'''For normalizing maybe 
    Data is already normalized in plot?'''
    
minH = dfA['Height'].min()
minW = dfA['Weight'].min()
maxH = dfA['Height'].max()
maxW = dfA['Weight'].max()
'''How to manipulate DF values now'''


'''Plotting the Data'''
plt.scatter(females['Weight'], females['Height'], color='r', marker='.')

plt.scatter(males['Weight'], males['Height'], color='b', marker='.')

plt.show()
'''Need to add groupB/C.txt data'''

