import tensorflow as tf
from tensorflow.contrib import slim
if __name__ == "__main__":
    from BiLSTM import BiLSTM
    from SelfAttention import SelfAttention
    from Dataset import Dataset
else:
    from .BiLSTM import BiLSTM
    from .SelfAttention import SelfAttention
    from .Dataset import Dataset

class SentenceRepretention(object):
    def __init__(self, lstm_size=128, layers=2, wv_dim=100, dim_a=200, dim_r=30, alpha=0.01, lr=0.01, norm=5.0, drop_out=0.5):
        x, vocab, dropout, output = BiLSTM(lstm_size=lstm_size, layers=layers, wv_dim=wv_dim).io_nodes()
        self._x = x
        self._vocab = vocab
        self._dropout = dropout
        self._output = output
        self._attention = SelfAttention(output, 2*lstm_size, dim_a=dim_a, dim_r=dim_r).get_attention()
        self._dim_r = dim_r
        self._lstm_size = lstm_size
        self._alpha = alpha
        self._lr = lr
        self._norm = norm
        self._dr = drop_out
        self._build_graph()                

    def _build_graph(self):            
        self._y = tf.placeholder(dtype=tf.int64, shape=[None], name='input-y')
        self._lens = tf.placeholder(dtype=tf.int64, shape=[None], name='input-lens')

        with tf.name_scope('sentence-embedding'):
            self._M = tf.matmul(self._attention, self._output)

        with tf.name_scope('fully-connected-layer'):
            sentence_embedding = tf.reshape(self._M, shape=[-1, 2*self._dim_r*self._lstm_size])
            self._fc = slim.fully_connected(sentence_embedding, 2, activation_fn=tf.nn.relu)
        
        with tf.name_scope('penalization'):
            AA_T = tf.matmul(self._attention, tf.transpose(self._attention, [0, 2, 1]))
            cur_batch = tf.shape(self._x)[0]
            I = tf.eye(self._dim_r, batch_shape=[cur_batch], dtype=tf.float64)
            P = tf.square(tf.norm(AA_T - I, axis=[1, 2], ord='fro'))
        
        with tf.name_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._fc, labels=self._y)
            self._loss = tf.reduce_mean(loss) + tf.reduce_mean(self._alpha*P)
            optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=self._norm)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=-self._norm)
            self._train_op = optimizer.minimize(self._loss)
            pass

    def fit(self, datapath, wv, batch_size=100, epoch=5):            
        with tf.Session() as sess:
            dataset = Dataset(sess, datapath, batch_size, '\t', epoch=epoch)
            sess.run(tf.initialize_all_variables())
            for steps, (c, ws, lens) in enumerate(dataset):
                feed = {self._x : ws, self._y: c, self._lens: lens, self._vocab: wv, self._dropout: self._dr}
                _, loss = sess.run([self._train_op, self._loss], feed_dict=feed)
                print('step %d, loss: %f' % (steps, loss))
          
if __name__ == "__main__":
    import numpy as np
    wv = np.random.random([500000, 100])
    SentenceRepretention().fit('../url-sms-train-samples.csv', wv)