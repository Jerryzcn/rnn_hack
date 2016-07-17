import pandas as pd
import pickle
from config import *


def getTopLoc():
    '''
    :return: data sorted by time
    '''
    #df = pd.read_csv('./data/uber-raw-data-janjune-15.csv',sep=',',nrows=1000)
    df = pd.read_csv('/home/hehe/projects/uber/data/uber-raw-data-janjune-15.csv',sep=',')

    # Dispatching_base_num Pickup_date Affiliated_base_num locationID
    data = df.values
    count = {}
    for i in range(len(data)):
        if data[i,3]-1 not in count:
            count[data[i,3]-1] = 0

        count[data[i,3]-1]+=1

    count = count.items()

    count.sort(key=lambda tup:tup[1], reverse=True)
    print(count)

    _sum = sum([tup[1] for tup in count])
    _t = 0
    candidates = []
    for i in range(len(count)):
        _t += count[i][1]
        if float(_t)/_sum > 0.3:
            break
        candidates.append(count[i][0])
        print('{} {} {}'.format(count[i][0], count[i][1], float(_t)/_sum))

    return candidates

candidates = getTopLoc()
print(len(candidates))
print(float(len(candidates))/LOCATION_ID_MAX)
pickle.dump(candidates, open('./data/top_loc.dat','wb'))
