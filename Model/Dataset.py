import tensorflow as tf

class Dataset(object):
    def __init__(self, sess, path, batch_size, sep='\t', epoch=5):
        self._path = path
        self._sess = sess
        self._sep = sep
        self._batch_size = batch_size
        self._cur_epoch = 0
        self._epoch = epoch
        self._build_graph()

    def _build_graph(self):
        dataset = tf.data.TextLineDataset(self._path)
        def __parse_line(line):
            items = tf.string_split([line], delimiter=self._sep).values
            c, ws = items[0], items[1]
            ws = tf.string_split([ws], delimiter=' ').values
            ws = tf.string_to_number(ws, tf.int32)
            c = tf.string_to_number(c, tf.int32)
            return c, ws, tf.size(ws)

        dataset = dataset.map(__parse_line)
        padded_shapes = (tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([]))
        self._batched_dataset = dataset.padded_batch(self._batch_size, padded_shapes=padded_shapes)

        self._iterator = self._batched_dataset.make_initializable_iterator()
        self._c_op, self._ws_op, self._len_op = self._iterator.get_next()

    def __next__(self):
        try:
            c, ws, lens = self._sess.run([self._c_op, self._ws_op, self._len_op])
            return c, ws, lens
        except tf.errors.OutOfRangeError:
            if self._cur_epoch >= self._epoch:
                raise StopIteration()
            else:
                self._sess.run(self._iterator.initializer)
                c, ws, lens = self._sess.run([self._c_op, self._ws_op, self._len_op])
                return c, ws, lens

    def __iter__(self):
        self._sess.run(self._iterator.initializer)
        return self