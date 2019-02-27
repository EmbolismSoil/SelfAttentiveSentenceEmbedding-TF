import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

class SelfAttention(object):
    def __init__(self, input, input_dim=256, dim_a=200, dim_r=10):
        self._input = input
        self._input_dim = input_dim
        self._dim_a = dim_a
        self._dim_r = dim_r
        self._build_graph()

    def _build_graph(self):
        initializer = xavier_initializer()
        shape = tf.shape(self._input)
        seq_len = shape[1]

        input_reshaped = tf.reshape(self._input, shape=[-1, self._input_dim])
        with tf.name_scope('self-attention'):
            self._Ws1 = tf.get_variable("Ws1", shape=[self._input_dim, self._dim_a], dtype=tf.float64, initializer=initializer)
            Hs_1 = tf.tanh(tf.matmul(input_reshaped, self._Ws1), name='Hs')
            self._Ws2 = tf.get_variable('Ws2', shape=[self._dim_a, self._dim_r], dtype=tf.float64, initializer=initializer)
            Hs_2 = tf.matmul(Hs_1, self._Ws2)
            Hs_2 = tf.reshape(Hs_2, [-1, seq_len, self._dim_r])
            Hs_2 = tf.transpose(Hs_2, [0, 2, 1])
            self._A = tf.nn.softmax(Hs_2, name='attention')

    def get_attention(self):
        return self._A

if __name__ == "__main__":
    from BiLSTM import BiLSTM
    x, dropout, output = BiLSTM().io_nodes()
    attention = SelfAttention(output).get_attention()