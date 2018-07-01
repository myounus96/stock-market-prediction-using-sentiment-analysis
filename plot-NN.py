import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1=pd.read_csv('NN-DATA-all.csv',usecols=['Y'])

df2=pd.read_csv('NN-DATA-all.csv',usecols=['Y-ENN'])

df3=pd.read_csv('NN-DATA-all.csv',usecols=['Y-ANN'])

data1 = df1.values.tolist()
data2 = df2.values.tolist()
data3 = df3.values.tolist()

plt.plot(range(705),data1, label='Y')
plt.plot(range(705),data2, label='Y-ENN')
plt.plot(range(705),data3, label='Y-ANN')

plt.xlabel('Row Number')
plt.ylabel('Values')

plt.title("NN Plot")

plt.legend()

plt.show()