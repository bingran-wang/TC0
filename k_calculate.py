import numpy as np

import csv
import pandas as pd
df = pd.read_excel (r'naca2412_input.xlsx')
alpha = df.values[:,0]
Cl = df.values[:,1]
Cd = df.values[:,2]
Cm = df.values[:,4]

Cl_min = -1.2869
#Cd = np.array([0.0066, 0.00548, 0.00906, 0.01471, 0.01825,0.00568,0.09506])
#Cl = np.array([-0.0003, 0.3217, 0.8784, 1.2032, 1.3684,0.2442,1.4781])
x = (Cl-Cl_min)**2
m,b = np.polyfit(x, Cd, 1)
print('K = ', m)