import numpy as np
from keras.optimizers import SGD
from scipy.spatial import distance
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Embedding, Flatten
from keras.regularizers import l2, activity_l2
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

class Model():

	#INITIALIZE
	def __init__(self, file, mode, identifier): 
		self.data_file = file
		self.mode = mode
		self.identifier = identifier
		self.xtrain = []
		self.ytrain = []
		self.xtest = []
		self.ytest = []
		self.vocab = set()
		self.maxlen = 0
		self.batchsize = 5

	#TRAINING DATA	
	def ingest_data(self):
		print 'Ingesting Training Data'

		df = self.data_file

		if len(df) == 4: 
			xtrain, ytrain, xtest, ytest = df
		
		elif len(df) == 2:
			x, y = df
			split = int(0.60*len(x))
			xtrain = x[:split]
			ytrain = y[:split]
			xtest  = x[split:]
			ytest = y[split: ]

		self.xtrain = xtrain 
		self.ytrain = ytrain	
		self.xtest = xtest 
		self.ytest = ytest 

		print 'Data Description: '
		print '\n> DATA DESC'
		print '\tSplit Data: ', len(xtrain), type(ytrain), '   |   ', len(xtest), type(ytest)
		

	def build(self):
		print '\nBuilding Model'
		xtrain = self.xtrain  
		ytrain = self.ytrain 
		xtest = self.xtest 
		ytest = self.ytest	
		xtrain = np.array(xtrain)
		ytrain = np.array(ytrain)

		print '------>', xtrain.shape , ytrain.shape, len(ytrain)
		print '------>', xtest.shape , ytest.shape, len(ytest)

		#HYPERPARAMETERS
		#input_shape == (1,28,28)/ (3,28,28)
		
		epochs = 25
		num_filters1 = 20
		k_size1 = 3
		num_filters2 = 32
		k_size2 = 5	
		num_classes = 265
				
		dropout = [0.2,0.4]
		hidden_dims1 = 500
		batch_size = 1
		i_shape = (1, 500, 500)

		#BuildModel
		model = Sequential()  
		model.add(Convolution2D(num_filters1, k_size1, k_size1, border_mode='valid', activation='relu', input_shape=i_shape))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Convolution2D(num_filters2, k_size2, k_size2, border_mode='valid', activation='relu')
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(dropout[0]))
		model.add(Flatten())
		model.add(Dense(hidden_dims1))
		model.add(Dense(265))
		model.add(Activation('softmax'))
		print (model.summary())
		#Compile Model
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(xtrain, ytrain,  batch_size=batch_size, nb_epoch=epochs, verbose=2)
		scores = model.evaluate(x_test, y_test, verbose=0)
		predict = model.predict(x_test, batch_size=batch_size)
		print("\n\n> ACCURACY: ", scores[1]*100)
		print predict
		evaluate(y_test, predict)



	def evaluate(self, actual, predicted):
		for i in range(len(predicted)):
			if predicted[i] >= 0.5: predicted[i] = 1
			else: predicted[i] = 0 
		confusion = confusion_matrix(actual, predicted)
		accuracy = accuracy_score(actual, predicted)
		print '\n> CONFUSION MATRIX: \n', confusion
		precision = confusion[1][1]/(confusion[1][1]+confusion[0][1])
		recall = confusion[1][1]/(confusion[1][1]+confusion[1][0])
		print '\n\t> Precision: ', precision 
		print '\n\t> Recall: ', recall 
		print '\n\t> F1: ', 2*precision*recall/(precision+recall)
		print '\n> ACCURACY: ', accuracy
		print '---------\n\n'
		print classification_report(actual, predicted)	



