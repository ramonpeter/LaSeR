import pandas as pd
import numpy as np

t = []

r1 = np.random.normal(4,1, [50000,1])
r2 = np.random.normal(4,1, [50000,1])
r3 = np.random.normal(4,1, [50000,1])
phi1 = np.random.uniform(0,2*np.pi, [50000,1])
phi2 = np.random.uniform(0,2*np.pi, [50000,1])
phi3 = np.random.uniform(0,2*np.pi, [50000,1])

x1 = r1 * np.cos(phi1) + 12
x2 = r2 * np.cos(phi2) - 12
x3 = r3 * np.cos(phi3) 
y1 = r1 * np.sin(phi1)
y2 = r2 * np.sin(phi2)
y3 = r3 * np.sin(phi3)

t1 = np.concatenate((x1,y1),-1)
t2 = np.concatenate((x2,y2),-1)
t3 = np.concatenate((x3,y3),-1)

t = np.concatenate((t1,t2, t3))
np.random.shuffle(t)

s = pd.HDFStore('3_rings.h5')
s.append('data', pd.DataFrame(t))
s.close()
