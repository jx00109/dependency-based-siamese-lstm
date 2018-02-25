# -*- coding:utf8 -*-

import tensorflow as tf


class LSTMRNN(object):
    def singleRNN(self, x, scope, cell='lstm', reuse=None):
        if cell == 'gru':
            with tf.variable_scope('grucell' + scope, reuse=reuse, dtype=tf.float64):
                used_cell = tf.contrib.rnn.GRUCell(self.hidden_neural_size, reuse=tf.get_variable_scope().reuse)

        else:
            with tf.variable_scope('lstmcell' + scope, reuse=reuse, dtype=tf.float64):
                used_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_neural_size, forget_bias=2.5, state_is_tuple=True,
                                                         reuse=tf.get_variable_scope().reuse)

        with tf.variable_scope('grucell_init_state' + scope, reuse=reuse, dtype=tf.float64):
            self.cell_init_state = used_cell.zero_state(self.batch_size, dtype=tf.float64)

        with tf.name_scope('RNN_' + scope), tf.variable_scope('RNN_' + scope, dtype=tf.float64):
            outs, _ = tf.nn.dynamic_rnn(used_cell, x, initial_state=self.cell_init_state, time_major=False,
                                        dtype=tf.float64)
        return outs

    def __init__(self, config, is_training=True):
        self.batch_size = tf.Variable(0, dtype=tf.int32, trainable=False)

        num_step = config.num_step
        embed_dim = config.embed_dim
        self.input_data_s1 = tf.placeholder(tf.float64, [None, num_step, embed_dim])
        self.input_data_s2 = tf.placeholder(tf.float64, [None, num_step, embed_dim])
        self.target = tf.placeholder(tf.float64, [None])
        self.mask_s1 = tf.placeholder(tf.float64, [None, num_step])
        self.mask_s2 = tf.placeholder(tf.float64, [None, num_step])

        self.hidden_neural_size = config.hidden_neural_size
        self.new_batch_size = tf.placeholder(tf.int32, shape=[], name="new_batch_size")
        self._batch_size_update = tf.assign(self.batch_size, self.new_batch_size)

        with tf.name_scope('lstm_output_layer'):
            self.cell_outputs1 = self.singleRNN(x=self.input_data_s1, scope='side1', cell='lstm', reuse=None)
            self.cell_outputs2 = self.singleRNN(x=self.input_data_s2, scope='side1', cell='lstm', reuse=True)

        with tf.name_scope('Sentence_Layer'):
            # self.sent1 = tf.reduce_sum(self.cell_outputs1 * self.mask_s1[:, :, None], axis=1)
            # self.sent2 = tf.reduce_sum(self.cell_outputs2 * self.mask_s2[:, :, None], axis=1)
            # self.mask_s1_sum=tf.reduce_sum(self.mask_s1,axis=0)
            # self.mask_s2_sum=tf.reduce_sum(self.mask_s2,axis=0)
            # self.mask_s1_sum1 = tf.reduce_sum(self.mask_s1, axis=1)
            # self.mask_s2_sum1 = tf.reduce_sum(self.mask_s2, axis=1)
            self.sent1 = tf.reduce_sum(self.cell_outputs1 * self.mask_s1[:, :, None], axis=1)
            self.sent2 = tf.reduce_sum(self.cell_outputs2 * self.mask_s2[:, :, None], axis=1)

        with tf.name_scope('loss'):
            diff = tf.abs(tf.subtract(self.sent1, self.sent2), name='err_l1')
            diff = tf.reduce_sum(diff, axis=1)
            self.sim = tf.clip_by_value(tf.exp(-1.0 * diff), 1e-7, 1.0 - 1e-7)
            self.loss = tf.square(tf.subtract(self.sim, tf.clip_by_value((self.target - 1.0) / 4.0, 1e-7, 1.0 - 1e-7)))

        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(self.loss)
            self.truecost = tf.reduce_mean(tf.square(tf.subtract(self.sim * 4.0 + 1.0, self.target)))

        if not is_training:
            return

        self.globle_step = tf.Variable(0, name="globle_step", trainable=False, dtype=tf.float64)
        self.lr = tf.Variable(0.0, trainable=False, dtype=tf.float64)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self.cost, tvars)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)

        with tf.name_scope('train'):
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.new_lr = tf.placeholder(tf.float64, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self.new_lr)

    def assign_new_lr(self, session, lr_value):
        lr, _ = session.run([self.lr, self._lr_update], feed_dict={self.new_lr: lr_value})
        return lr

    def assign_new_batch_size(self, session, batch_size_value):
        session.run(self._batch_size_update, feed_dict={self.new_batch_size: batch_size_value})
