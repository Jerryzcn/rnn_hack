from preprocess import getData
import pickle

#intervals = [15,30,60]
intervals = [60]
for i in intervals:
    data, data_unnormalized = getData(i)
    # pickle.dump(data,open(str(i)+".p","wb"))
    pickle.dump(data_unnormalized,open(str(i)+"_unnormalized.p","wb"))