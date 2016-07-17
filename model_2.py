import tensorflow as tf
from config import *
import numpy as np
from summary import Summary

class Model:
    def __init__(self):

        self.x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
        self.y = tf.placeholder(tf.float32, [None, HIDDEN_3_SIZE])

        W_1 = tf.Variable(tf.random_uniform([NUM_FEATURES, HIDDEN_1_SIZE], maxval=1.0))
        b_1 = tf.Variable(tf.random_uniform([HIDDEN_1_SIZE], maxval=1.0))

        W_2 = tf.Variable(tf.random_uniform([HIDDEN_1_SIZE, HIDDEN_2_SIZE], maxval=1.0))
        b_2 = tf.Variable(tf.random_uniform([HIDDEN_2_SIZE], maxval=1.0))

        W_3 = tf.Variable(tf.random_uniform([HIDDEN_2_SIZE, HIDDEN_3_SIZE], maxval=1.0))
        b_3 = tf.Variable(tf.random_uniform([HIDDEN_3_SIZE], maxval=1.0))

        x_drop = tf.nn.dropout(self.x, KEEP_PROB_INPUT)

        h_1 = tf.nn.tanh(tf.matmul(x_drop, W_1) + b_1)
        h_1_drop = tf.nn.dropout(h_1, KEEP_PROB_HIDDEN)

        h_2 = tf.nn.tanh(tf.matmul(h_1_drop, W_2) + b_2)
        h_2_drop = tf.nn.dropout(h_2, KEEP_PROB_HIDDEN)

        h_3 = tf.matmul(h_2_drop, W_3) + b_3

        # self.y_pred = tf.nn.softmax(h_3)
        self.y_pred = h_3

        # self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.y_pred), reduction_indices=[1]))
        self.cross_entropy = tf.reduce_mean(tf.square(self.y_pred - self.y))

        self.train_step = tf.train.MomentumOptimizer(109,0.99).minimize(self.cross_entropy)

        self.sess = tf.Session()

    def fit(self, X, Y, valid_X = None, valid_Y = None):
        self.sess.run(tf.initialize_all_variables())
        summary_train = Summary('train_loss')
        summary_valid = Summary('valid_loss')
        summary_valid_acc = Summary('valid_acc')
        summary_train.start()
        summary_valid.start()
        summary_valid_acc.start()

        for iter in range(NUM_ITER):
            total_loss = 0.0
            for i in range(0,len(X), BATCH_SIZE):
                tail = min(i+BATCH_SIZE, len(X))
                batch_xs = X[i:tail,:]
                batch_ys = Y[i:tail,:]
                _, loss = self.sess.run([self.train_step, self.cross_entropy], feed_dict={self.x: batch_xs, self.y: batch_ys})
                total_loss += loss*(tail-i+1)
            total_loss /= float(len(X))
            print('Avg training loss on {}th epoch: {}'.format(iter, total_loss))
            summary_train.log(total_loss)

            if valid_X is not None and valid_Y is not None and iter % 100 ==0:
                valid_loss, valid_acc = self.eva(valid_X, valid_Y)
                summary_valid.log(valid_loss)
                summary_valid_acc.log(valid_acc)


    def eva(self, X, Y):

        total_loss = 0.0
        top_10_percentage = 0.0

        for i in range(0,len(X), BATCH_SIZE):
                tail = min(i+BATCH_SIZE, len(X))
                batch_xs = X[i:tail,:]
                batch_ys = Y[i:tail,:]
                y_pred = self.sess.run(self.y_pred, feed_dict={self.x: batch_xs})
                total_loss += np.sum(np.square(batch_ys-y_pred))

        total_loss /=float(len(X))
        top_10_percentage = (top_10_percentage*100)/ float(len(X))
        print('Evaluation loss: {} Top 10: {}%'.format(total_loss, top_10_percentage))
        return total_loss, top_10_percentage


    def predict(self, X):
        rv = None
        for i in range(0,len(X), BATCH_SIZE):
                tail = min(i+BATCH_SIZE, len(X))
                batch_xs = X[i:tail,:]
                y_pred = self.sess.run(self.y_pred, feed_dict={self.x: batch_xs})
                if rv is None:
                    rv = y_pred
                else:
                    rv = np.concatenate((rv, y_pred))
        return rv




