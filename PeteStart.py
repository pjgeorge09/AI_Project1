import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('groupA.txt', header=None, names = ['Height', 'Weight', 'Gender'])
print(df)

'''Splits the original bulk data into two'''
'''https://stackoverflow.com/questions/27900733/python-pandas-separate-a-dataframe-based-on-a-column-value'''
males = df[df.Gender == 0]
females = df[df.Gender == 1]

X = males["Height"].tolist() '''maybe use'''

'''currently as 2 separate plots'''
males.plot(kind='scatter',x='Weight',y='Height',color='blue')
females.plot(kind='scatter',x='Weight',y='Height',color='red')


'''Need to NORMALIZE data!'''