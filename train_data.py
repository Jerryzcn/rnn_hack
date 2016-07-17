

from datetime import datetime
import numpy as np
from sklearn.utils import shuffle
from config import *
import pickle


def getFeatures(meta):
    features = []
    for i in range(len(meta)):
        features.append([0]*NUM_FEATURES)
        features[i][meta[i]['interval']] = 1

        date_time = datetime.strptime(meta[i]['date'], '%Y-%m-%d')

        features[i][NUM_INTERVALS+date_time.weekday()] = 1
    return features

def getFeatures_single(meta, loc2id):
    features = []
    for i in range(len(meta)):
        features.append([0]*NUM_FEATURES)
        features[i][meta[i]['interval']] = 1

        date_time = datetime.strptime(meta[i]['date'], '%Y-%m-%d')

        features[i][NUM_INTERVALS+date_time.weekday()] = 1
        # features[i][NUM_INTERVALS+NUM_DAY_OF_WEEK+date_time.da] = 1
    return features

def getTrainTestData(loc = None):
    data = pickle.load(open('./data/60_unnormalized.p', "rb"))

    raw_meta = []
    raw_data = []
    for k,v in data.iteritems():
        for i in range(len(v)):

            _d = []
            for j in range(len(loc)):
                _d.append(v[i][loc[j]])


            _sum = sum(_d)
            if _sum == 0:
                continue
            _d = [float(item)/_sum for item in _d]


            raw_meta.append({"date":k,"interval":i})
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

def getTrainTestData_single(loc = None):
    data = pickle.load(open('./data/60.p', "rb"))
    loc2id = {}
    for i in range(len(loc)):
        loc2id[loc[i]]=i
    raw_meta = []
    raw_data = []
    for k,v in data.iteritems():
        for i in range(len(v)):
            for j in range(len(loc)):
                _d = []
                _d.append(v[i][loc[j]])
                raw_meta.append({"date":k,"interval":i,'loc':loc[j]})
                raw_data.append(_d)



    num = len(raw_data)

    train_meta_data = raw_meta[0:int(0.6*num)]
    valid_meta_data = raw_meta[int(0.6*num):int(0.8*num)]
    test_meta_data = raw_meta[int(0.8*num):]

    train_y = raw_data[0:int(0.6*num)]
    valid_y = raw_data[int(0.6*num):int(0.8*num)]
    test_y = raw_data[int(0.8*num):]

    train_X = getFeatures_single(train_meta_data, loc2id)
    valid_X = getFeatures_single(valid_meta_data, loc2id)
    test_X = getFeatures_single(test_meta_data, loc2id)

    train_X = np.array(train_X, dtype=np.float32)
    valid_X = np.array(train_X, dtype=np.float32)
    test_X = np.array(train_X, dtype=np.float32)

    train_y = np.array(train_y, dtype=np.float32)
    valid_y = np.array(train_y, dtype=np.float32)
    test_y = np.array(train_y, dtype=np.float32)

    train_X, train_y = shuffle(train_X, train_y, random_state=0)
    valid_X, valid_y = shuffle(valid_X, valid_y, random_state=1)
    test_X, test_y = shuffle(test_X, test_y, random_state=2)

    return train_X, train_y, valid_X, valid_y, test_X, test_y

def getTrainTestData_single(loc = None):
    data = pickle.load(open('./data/60.p', "rb"))
    loc2id = {}
    for i in range(len(loc)):
        loc2id[loc[i]]=i
    raw_meta = []
    raw_data = []
    for k,v in data.iteritems():
        for i in range(len(v)):
            for j in range(len(loc)):
                # _d = []
                # _d.append([v[i][loc[j]],1.0-v[i][loc[j]]])
                _d = [v[i][loc[j]],1.0-v[i][loc[j]]]
                raw_meta.append({"date":k,"interval":i,'loc':loc[j]})
                raw_data.append(_d)



    num = len(raw_data)

    train_meta_data = raw_meta[0:int(0.6*num)]
    valid_meta_data = raw_meta[int(0.6*num):int(0.8*num)]
    test_meta_data = raw_meta[int(0.8*num):]

    train_y = raw_data[0:int(0.6*num)]
    valid_y = raw_data[int(0.6*num):int(0.8*num)]
    test_y = raw_data[int(0.8*num):]

    train_X = getFeatures_single(train_meta_data, loc2id)
    valid_X = getFeatures_single(valid_meta_data, loc2id)
    test_X = getFeatures_single(test_meta_data, loc2id)

    train_X = np.array(train_X, dtype=np.float32)
    valid_X = np.array(train_X, dtype=np.float32)
    test_X = np.array(train_X, dtype=np.float32)

    train_y = np.array(train_y, dtype=np.float32)
    valid_y = np.array(train_y, dtype=np.float32)
    test_y = np.array(train_y, dtype=np.float32)

    train_X, train_y = shuffle(train_X, train_y, random_state=0)
    valid_X, valid_y = shuffle(valid_X, valid_y, random_state=1)
    test_X, test_y = shuffle(test_X, test_y, random_state=2)

    return train_X, train_y, valid_X, valid_y, test_X, test_y