from jax import numpy as np
import os

#os.environ["TF_CPP_MIN_LOG_LEVEL"]=0

a = np.array([2, 3, 4])
b = np.array([3, 4, 5])

print(a + b)