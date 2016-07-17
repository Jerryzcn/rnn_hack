import os
import sys
import pickle
from one import *
from model import *




#python main.py training_ready(0/1) model_ready(0,1) identifier
training_ready = int(sys.argv[1])
model_ready = int(sys.argv[2])
if len(sys.argv) == 3: identifier = ''
elif len(sys.argv) == 4: identifier = sys.argv[3]

#default:
# training_ready = 1
model_ready = 0
identifier = ''


##DATA:
if training_ready == 0:
	#GetFiles
	file_lst = get_files()
	#Convert To Numpy
	data = convt_img(file_lst)
	#Download_Data
	with open('pickled_data.txt', 'wb') as file: pickle.dump(data,file)
else: 
	with open('pickled_df2.txt', 'rb') as file: df = pickle.load(file)

# print 'data: ', len(df)
# print 'train: ', len(df[0]), df[0].shape, df[0][0].shape

##MODEL:
convnet =  Model(df, model_ready, identifier)

#If model is to be trained, get input data, build model, save model and store the identifier
if model_ready == 0:
	convnet.ingest_data()
	convnet.build()
    # identifier = model.get_identifier()
# elif model_ready == 1:
	