import pandas as pd
import numpy as np

t = []
for i in range(2):
	for j in range(2):
		t.append(np.random.multivariate_normal([-4 + 8 * i, -4 + 8 * j], [[1,0],[0,1]], 25000))

t = np.concatenate(t)
np.random.shuffle(t)

s = pd.HDFStore('4_gaussians.h5')
s.append('data', pd.DataFrame(t))
s.close()
