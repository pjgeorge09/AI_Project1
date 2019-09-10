import pandas as pd

xMale = []
yMale = []
xFemale = []
yFemale = []
z = []

df = pd.read_csv('groupA.txt', header=None, names = ['Height', 'Weight', 'Gender'])
print(df)

'''Puts ALL values into xMale right now'''
if (df['Gender'] == '0'):
    xMale = df['Height'].tolist()
print(xMale)
'''https://stackoverflow.com/questions/53979790/reading-columns-from-csv-with-no-labels-in-pandas'''

'''df.['Height'].tolist()'''
for row in df:
    if row.endswith("0"):
        xMale.append(int(row[0]))
        yMale.append(int(row[1]))

