import tensorflow as tf
from Model import SentencePresentation
import numpy as np
from gensim.models import word2vec
import json
from Model.Dataset import Dataset

if __name__ == "__main__":
    model = word2vec.Word2Vec.load('./wv.gensim.model')
    wv = model.wv.vectors
    network =SentencePresentation(wv, wv_dim=100, lstm_size=64, layers=1, dim_r=30, classes=4, dim_a=10, norm=0.5, lr=0.01)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for steps, loss, acc in network.fit(sess, './train_data.csv', epoch=2):
            print('steps: %d, loss: %f, acc: %f' % (steps, loss, acc))

        dataset = Dataset(sess, './train_data.csv', 200, '\t', max_len=500, epoch=1)
        for c, ws, lens in dataset:
            labels, attentions, embedding = network.predict(sess, ws, lens)
            for w, attention in zip(ws, attentions):
                words_json = []
                w = [model.wv.index2word[i] for i in w]
                at = np.max(attention, axis=0)
                for c, a in zip(w, at):
                    words_json.append({'word': c, 'attention': a})
                words_json.append({'word': '\n', 'attention': 0.0})
                print('\n\n')
                print(json.dumps(words_json, ensure_ascii=False))
