import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd .read_csv('glass.csv')

print('Target variables  : ', dataset['target_names'])

(unique, counts) = np.unique(dataset['target'], return_counts=True)

print('Unique values of the target variable', unique)
print('Counts of the target variable :', counts)

plt.title('Target variable counts in dataset')
plt.show()