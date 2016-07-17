

import datetime
import numpy as np
from sklearn.utils import shuffle
from config import *
import pickle


def getFeatures(meta):
    features = []
    for i in range(len(meta)):
        features.append([0]*(NUM_FEATURES-2*LOCATION_ID_MAX))
        features[i][meta[i]['interval']] = 1

        date_time = datetime.datetime.strptime(meta[i]['date'], '%Y-%m-%d')

        features[i][NUM_INTERVALS+date_time.weekday()] = 1
        features[i]+=meta[i]['previous'][0]
        features[i]+=meta[i]['previous'][1]

    return features


def getTrainTestData():
    data = pickle.load(open('./data/60_unnormalized.p', "rb"))

    raw_meta = []
    raw_data = []
    for k,v in data.iteritems():
        for i in range(len(v)):

            _d = v[i]
            previous = [[0]*LOCATION_ID_MAX,[0]*LOCATION_ID_MAX]
            if i==0:
                # previous date
                date_time = datetime.datetime.strptime(k, '%Y-%m-%d')
                previous_day = date_time - datetime.timedelta(1)
                str_previous_day = previous_day.strftime('%Y-%m-%d')
                if str_previous_day in data:
                    previous[0]=data[str_previous_day][-2]
                    previous[1]=data[str_previous_day][-1]
            elif i==1:
                # previous date
                date_time = datetime.datetime.strptime(k, '%Y-%m-%d')
                previous_day = date_time - datetime.timedelta(1)
                str_previous_day = previous_day.strftime('%Y-%m-%d')
                previous[1]=v[i-1]
                if str_previous_day in data:
                    previous[0]=data[str_previous_day][-1]
            else:
                previous[0]=v[i-2]
                previous[1]=v[i-1]

            raw_meta.append({"date":k,"interval":i,"previous":previous})
            raw_data.append(_d)

    num = len(raw_data)

    train_meta_data = raw_meta[0:int(0.6*num)]
    valid_meta_data = raw_meta[int(0.6*num):int(0.8*num)]
    test_meta_data = raw_meta[int(0.8*num):]

    train_y = raw_data[0:int(0.6*num)]
    valid_y = raw_data[int(0.6*num):int(0.8*num)]
    test_y = raw_data[int(0.8*num):]

    train_X = getFeatures(train_meta_data)
    valid_X = getFeatures(valid_meta_data)
    test_X = getFeatures(test_meta_data)

    train_X = np.array(train_X, dtype=np.float32)
    valid_X = np.array(valid_X, dtype=np.float32)
    test_X = np.array(test_X, dtype=np.float32)

    train_y = np.array(train_y, dtype=np.float32)
    valid_y = np.array(valid_y, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.float32)

    train_X, train_y = shuffle(train_X, train_y, random_state=0)
    valid_X, valid_y = shuffle(valid_X, valid_y, random_state=1)
    test_X, test_y = shuffle(test_X, test_y, random_state=2)

    return train_X, train_y, valid_X, valid_y, test_X, test_y

