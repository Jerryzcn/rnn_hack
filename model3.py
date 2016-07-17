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

        self.y_pred = tf.nn.softmax(h_3)
        # self.y_pred = h_3
        self._cross_entropy = tf.reduce_sum(-tf.reduce_sum(self.y * tf.log(self.y_pred), reduction_indices=[1]))
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.y_pred), reduction_indices=[1]))
        #self.cross_entropy = tf.reduce_mean(tf.square(self.y_pred - self.y))

        self.train_step = tf.train.AdagradOptimizer(0.01).minimize(self.cross_entropy)

        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_pred,1))
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"))

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
                _, loss = self.sess.run([self.train_step, self._cross_entropy], feed_dict={self.x: batch_xs, self.y: batch_ys})
                total_loss += loss
            total_loss /= float(len(X))
            print('Avg training loss on {}th epoch: {}'.format(iter, total_loss))
            summary_train.log(total_loss)

            if valid_X is not None and valid_Y is not None and iter % 100 ==0:
                valid_loss, valid_acc = self.eva(valid_X, valid_Y)
                summary_valid.log(valid_loss)
                summary_valid_acc.log(valid_acc)


    def eva(self, X, Y):

        total_loss = 0.0
        total_acc = 0.0

        for i in range(0,len(X), BATCH_SIZE):
                tail = min(i+BATCH_SIZE, len(X))
                batch_xs = X[i:tail,:]
                batch_ys = Y[i:tail,:]
                loss, acc = self.sess.run([self._cross_entropy, self.accuracy], feed_dict={self.x: batch_xs, self.y: batch_ys})
                total_loss += loss
                total_acc += acc


        total_loss /=float(len(X))
        total_acc = (total_acc*100)/ float(len(X))
        print('Evaluation loss: {} Top 10: {}%'.format(total_loss, total_acc))
        return total_loss, total_acc


    def predict(self, X):
        rv = None
        for i in range(0,len(X), BATCH_SIZE):
                tail = min(i+BATCH_SIZE, len(X))
                batch_xs = X[i:tail,:]
                y_pred = self.sess.run(self.y_pred, feed_dict={self.x: batch_xs})
                if rv is None:
                    rv = np.argmax(y_pred, axis=1)
                else:
                    rv = np.concatenate((rv, np.argmax(y_pred, axis=1)))
        return rv




