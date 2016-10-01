import pickle
import os
from collections import defaultdict
path = 'data'
filenames = next(os.walk('data'))[2]
pAvg = defaultdict(int)

for pick in filenames:
    pick = pickle.load(open(path+'/'+pick, "rb"))
    for k,v in pick.items():
        pAvg[k] += pick[k]

with open('master_1000.txt', 'wb') as f:
           pickle.dump(pAvg,f)


