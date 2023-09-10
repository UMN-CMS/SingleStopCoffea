from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import pickle as pkl
import numpy as np
import hist 


data = pkl.load(open('updatedqcd.pkl', 'rb'))
m3 = data["histograms"]["m14_m"]['QCDInclusive2018',...]
print(m3)
