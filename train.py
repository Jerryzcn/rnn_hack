from train_data3 import getTrainTestData
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import pickle
from config import *
from sklearn.metrics import confusion_matrix

def week():
    X=[]
    for i in range(7):
        for t in range(24):
            d = [0.0]*NUM_FEATURES
            d[t]=1.0
            d[NUM_INTERVALS+i]=1.0
            # d[NUM_INTERVALS+NUM_DAY_OF_WEEK+loc]=1.0
            X.append(d)
    return np.array(X)

loc = pickle.load(open('./data/top_loc.dat','rb'))
loc = [loc[0]]
train_X, train_y, valid_X, valid_y, test_X, test_y = getTrainTestData()
print('train: {} valid: {}, test: {}'.format(len(train_X), len(valid_X), len(test_X)))
m = Model()
m.fit(train_X, train_y, valid_X, valid_y)
loss, acc = m.eva(test_X, test_y)
print('test: {}'.format(loss))

'''
X_week = week()
Y_week = m.predict(X_week)
visual_week = [Y_week[i] for i in range(len(Y_week))]
plt.plot(visual_week,'-')
plt.show()
'''
