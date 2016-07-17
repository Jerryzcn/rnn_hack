
import pickle
import matplotlib.pyplot as plt
import numpy as np

from config import *
raw_data = pickle.load(open('./data/60.p', "rb"))
data = raw_data.items()

loc= pickle.load(open('./data/top_loc.dat','rb'))

'''
for location in loc:
    x = []
    y = []
    sum = 0
    for i in range(len(data)):
        for t in range(0,24):
            x.append(data[i][1][t][location])
            sum+=data[i][1][t][location]
    print('{} {}'.format(location,sum))
    plt.plot(x,'-')
    plt.show()
'''

x = []
count={}
for i in range(len(data)):
    for t in range(0,24):
        idx = np.argmax(np.array(data[i][1][t]))
        x.append(idx)
        if idx not in count:
            count[idx]=0
        count[idx]+=1

plt.plot(x,'-')
plt.show()

count = count.items()
count.sort(key = lambda tup:tup[1], reverse=True)
print(count)