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

class SentencePresentation(object):
    def __init__(self, wv, lstm_size=128, layers=2, wv_dim=100, dim_a=200, dim_r=50, 
                    alpha=1.0, lr=0.001, norm=5.0, drop_out=0.5, classes=2):
        x, lens, vocab, dropout, output = BiLSTM(lstm_size=lstm_size, layers=layers, wv_dim=wv_dim).io_nodes()
        self._x = x
        self._lens = lens
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
        self._classes = classes
        self._wv = wv
        self._build_graph()                

    def _build_graph(self):            
        self._y = tf.placeholder(dtype=tf.int64, shape=[None], name='input-y')
        self._mask = tf.sequence_mask(self._lens, dtype=tf.float64)

        with tf.name_scope('sentence-embedding'):
            self._M = tf.matmul(self._attention, self._output)

        with tf.name_scope('fully-connected-layer'):
            self._sentence_embedding = tf.reshape(self._M, shape=[-1, 2*self._dim_r*self._lstm_size])
            #self._fc = slim.fully_connected(sentence_embedding, 500, activation_fn=tf.nn.relu)
            self._fc = slim.fully_connected(self._sentence_embedding, self._classes, activation_fn=None)
            self._pre = tf.nn.softmax(self._fc)
        
        with tf.name_scope('penalization'):
            AA_T = tf.matmul(self._attention, tf.transpose(self._attention, [0, 2, 1]))
            cur_batch = tf.shape(self._x)[0]
            I = tf.eye(self._dim_r, batch_shape=[cur_batch], dtype=tf.float64)
            P = tf.square(tf.norm(AA_T - I, axis=[1, 2], ord='fro'))
        
        with tf.name_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._fc, labels=self._y)
            
            self._loss = tf.reduce_mean(loss) + tf.reduce_mean(self._alpha*P)
            optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
            #optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=self._norm)
            #optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=-self._norm)
            self._train_op = optimizer.minimize(self._loss)
        
        with tf.name_scope('acc'):
            pre = tf.argmax(self._fc, axis=1)
            acc = tf.equal(pre, self._y)
            self._acc = tf.reduce_mean(tf.cast(acc, tf.float64))

    def fit(self, sess, datapath,batch_size=50, epoch=5, max_len=500):            
        dataset = Dataset(sess, datapath, batch_size, '\t', max_len=max_len, epoch=epoch)
        sess.run(tf.initialize_all_variables())
        for steps, (c, ws, lens) in enumerate(dataset):
            feed = {self._x : ws, self._y: c, self._lens: lens, self._vocab: self._wv, self._dropout: self._dr}
            loss, acc, _ = sess.run([self._loss, self._acc, self._train_op], feed_dict=feed)
            print('step %d,acc: %f, loss: %f' % (steps, acc, loss))
        pass

    def predict(self, sess, x, lens):
        feed = {self._x : x, self._vocab: self._wv, self._dropout: 1.0, self._lens: lens}
        labels, attentions, embedding = sess.run([self._pre, self._attention, self._sentence_embedding], feed_dict=feed)
        return labels, attentions, embedding