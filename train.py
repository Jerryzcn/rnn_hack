from train_data import getTrainTestData
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import pickle

def week():
    X=[]
    for i in range(7):
        for t in range(24):
            d = [0.0]*31
            d[t]=1.0
            d[24+i]=1.0
            X.append(d)
    return np.array(X)

loc = pickle.load(open('./data/top_loc.dat','rb'))
train_X, train_y, valid_X, valid_y, test_X, test_y = getTrainTestData(loc)
print('train: {} valid: {}, test: {}'.format(len(train_X), len(valid_X), len(test_X)))
m = Model()
m.fit(train_X, train_y, valid_X, valid_y)

X_week = week()
Y_week = m.predict(X_week)
visual_week = [Y_week[i,10]*100.0 for i in range(len(Y_week))]
plt.plot(visual_week,'-')
plt.show()

