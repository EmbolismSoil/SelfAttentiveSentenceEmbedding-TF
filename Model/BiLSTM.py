import tensorflow as tf
from tensorflow.nn.rnn_cell import BasicLSTMCell, DropoutWrapper, MultiRNNCell


class BiLSTM(object):
    def __init__(self, lstm_size=128, layers=2, wv_dim=100):
        self._lstm_size = lstm_size
        self._layers = layers
        self._wv_dim = wv_dim
        self._build_graph()

    def _build_graph(self):
        with tf.name_scope('input'):
            self._lens = tf.placeholder(dtype=tf.int64, shape=[None], name='input-lens')
            self._x = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input-x')
            self._vocab = tf.placeholder(dtype=tf.float64, shape=[None, self._wv_dim], name='input-word-vector')
            embedded_x = tf.nn.embedding_lookup(self._vocab, self._x)

        with tf.name_scope('multi-bilstm-layer'):
            self._drop_out_placeholder = tf.placeholder(tf.float64, name='drop_out')
            fw_cells = [DropoutWrapper(BasicLSTMCell(self._lstm_size, state_is_tuple=True),
                                       output_keep_prob=self._drop_out_placeholder) for _ in range(self._layers)]
            bw_cells = [DropoutWrapper(BasicLSTMCell(self._lstm_size, state_is_tuple=True),
                                       output_keep_prob=self._drop_out_placeholder) for _ in range(self._layers)]
            self._fw_cell = MultiRNNCell(fw_cells)
            self._bw_cell = MultiRNNCell(bw_cells)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                self._fw_cell, self._bw_cell, embedded_x, sequence_length=self._lens, dtype=tf.float64)
            output = tf.concat(outputs, axis=2)
            self._output = output

    def io_nodes(self):
        return self._x, self._lens, self._vocab, self._drop_out_placeholder, self._output


if __name__ == "__main__":
    lstm = BiLSTM()
