import tensorflow as tf
from Model import SentencePresentation
import numpy as np
from gensim.models import word2vec
import json
import os

if __name__ == "__main__":    
    wv = np.load('../miniwv.npy')
    with tf.device('/device:GPU:0'):
        network =SentencePresentation(wv, wv_dim=100, lstm_size=128, layers=2, dim_r=50, classes=2, dim_a=300, norm=0.5, lr=0.01)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        for steps, loss, acc in network.fit(sess, '../samples-train.csv', epoch=10, batch_size=200):
            print('step %d, loss: %f, acc: %f' % (steps, loss, acc))
            if steps > 1 and steps % 500 == 0:
                saver = tf.train.Saver()
                saver.save(sess, './model/bilstm-model', steps)
