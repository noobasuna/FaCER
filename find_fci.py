import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz, chisq
import os

def find_fci(data):
    G, edges = fci(data, fisherz, 0.05, verbose=False)
    return G, edges

data = np.genfromtxt(os.path.join('/home/tpei0009/GazeTR/attr_csv/all/mpii_xgaze_dist.csv'), delimiter=',',dtype=float, encoding=None)
data = data[1:, 1:]
print(data.shape)
output_graph, edges = find_fci(data)
with open((f'./fci_mpiiethx.txt'), 'w') as f:
    f.write(f'Csvpath fci_mpiiethx: Results {output_graph}\n')

