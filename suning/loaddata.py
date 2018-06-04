from scipy.io import arff
import scipy
import numpy as np
import pandas as pd

from skmultilearn import problem_transform

f = open("../suning_data/assemble.arff",encoding="utf-8")
counter = 0
while 1:
    temp = f.readline()
    counter += 1
    if counter==100000:
        break
    print(temp)
#df = pd.DataFrame(data)
