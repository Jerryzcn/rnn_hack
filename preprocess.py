import pandas as pd
from datetime import datetime
# 1-index based
LOCATION_ID_MAX =265
def getData(TIME_INTERVAL = 60):
    '''
    :return: data sorted by time
    '''
    #df = pd.read_csv('./data/uber-raw-data-janjune-15.csv',sep=',',nrows=1000)
    df = pd.read_csv('./data/uber-raw-data-janjune-15.csv',sep=',')
    # Dispatching_base_num Pickup_date Affiliated_base_num locationID
    data = df.values
    data = data[data[:,1].argsort()]
    dataBySlots = {}
    for i in range(len(data)):
        date_time = datetime.strptime(data[i,1], '%Y-%m-%d %H:%M:%S')
        date = data[i,1].split()[0]
        if date not in dataBySlots:
            dataBySlots[date] = [[] for i in range(1440/TIME_INTERVAL)]
        else:
            slot_id = (date_time.hour*60+date_time.minute) / TIME_INTERVAL
            dataBySlots[date][slot_id].append(data[i])

    dataBySlotsByLocation = {}
    dataBySlotsByLocation_unnormalized = {}
    # normalization
    for k,v in dataBySlots.iteritems():

        dataBySlotsByLocation[k] = []
        dataBySlotsByLocation_unnormalized[k] = []
        for slot in v:
            # count by locationID
            count = [0]*LOCATION_ID_MAX
            for d in slot:
                count[d[3]-1] += 1
            _sum = sum(count)

            if _sum != 0:
                # avoid divide by 0
                dataBySlotsByLocation[k].append([float(count[i])/_sum for i in range(len(count))])
            else:
                dataBySlotsByLocation[k].append(count)

            dataBySlotsByLocation_unnormalized[k].append(count)
    return dataBySlotsByLocation, dataBySlotsByLocation_unnormalized
