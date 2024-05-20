import numpy as np
import scipy.io

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

mat_data = scipy.io.loadmat('agents/phydrl/envs/matlab/matlab.mat')

print(mat_data['P'])