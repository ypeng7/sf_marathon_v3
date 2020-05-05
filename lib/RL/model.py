from RL.base_model import TfBaseConfig, TfBaseModel
import tensorflow as tf
import numpy as np

class QNConfig(TfBaseConfig):
    def __init__(self, model_save_path):
        super(QNConfig, self).__init__(model_save_path, keep_prob=1.0)

class SimpleCNN(TfBaseModel):
    def __init__(self, model_config, global_reuse, *args, graph=None, session=None, **kwargs):
        super(SimpleCNN, self).__init__(model_config, global_reuse, *args, graph=graph, sess=session, **kwargs)

    def build(self, *args, **kwargs):
        dqn_sizes = args[0]
        scope_name = args[1]
        width, height, channel, n_actions = dqn_sizes[0], dqn_sizes[1], dqn_sizes[2], dqn_sizes[3]

        self.state_inp = tf.placeholder(tf.float32, shape=[None, height, width, channel], name="state")
        self.inventory = tf.placeholder(tf.float32, shape=[None, 10])
        self.target_Q = tf.placeholder(tf.float32, shape=[None], name="tgtQ")
        self.actions_taken = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        with tf.variable_scope(scope_name):
            kernel1 = tf.get_variable("kernel1", initializer=tf.random_normal([5, 5, 5, 8], stddev=1/25.0))
            feature1 = tf.nn.conv2d(self.state_inp, kernel1, strides=[1, 2, 2, 1], padding="VALID")
            hidden1 = tf.nn.leaky_relu(feature1)

            kernel2 = tf.get_variable("kernel2", initializer=tf.random_normal([3, 3, 8, 12], stddev=1/10.0))
            feature2 = tf.nn.conv2d(hidden1, kernel2, [1, 1, 1, 1], padding="VALID")
            hidden2 = tf.nn.leaky_relu(feature2)

            kernel3 = tf.get_variable("kernel3", initializer=tf.random_normal([3, 3, 12, 16], stddev=1/10.0))
            feature3 = tf.nn.conv2d(hidden2, kernel3, [1, 1, 1, 1], padding="VALID")
            hidden3 = tf.nn.leaky_relu(feature3)

            fc_inp_cnn = tf.reshape(hidden3, [-1, (hidden3.shape[1] * hidden3.shape[2] * hidden3.shape[3]).value])
            fc_inp = tf.concat([fc_inp_cnn, self.inventory], axis=1)

            fc_w1 = tf.get_variable("fc_w1", initializer=tf.random_normal([fc_inp.shape[1].value, fc_inp.shape[1].value // 5], stddev=1/300.0))
            fc_b1 = tf.get_variable("fc_b1", initializer=tf.random_normal([fc_inp.shape[1].value // 5], stddev=1/100.0))
            fc_hidden1 = tf.nn.leaky_relu(tf.matmul(fc_inp, fc_w1) + fc_b1)

            fc_w2 = tf.get_variable("fc_w2", initializer=tf.random_normal([fc_hidden1.shape[1].value, n_actions], stddev=1/150.0))
            fc_b2 = tf.get_variable("fc_b2", initializer=tf.random_normal([n_actions], stddev=1/100.0))
        self.Qsa = tf.matmul(fc_hidden1, fc_w2) + fc_b2

        oh_mask = tf.one_hot(self.actions_taken, depth=n_actions)
        self.oh_qsa = self.Qsa * oh_mask
        self.action_Qs = tf.reduce_sum(self.oh_qsa, axis=1)

        self.q_loss = tf.reduce_mean(tf.square((self.action_Qs - self.target_Q)))
        self.back_propagate(self.q_loss)

    def update(self, s, inv, a, y):
        feed_dict = {self.state_inp: s, self.inventory:inv, self.target_Q:y, self.actions_taken:a}
        _, loss, action_qs, qsa, oh_qsa = self.session.run([self.eval_op, self.q_loss, self.action_Qs, self.Qsa, self.oh_qsa], feed_dict=feed_dict)
        return loss, action_qs

    def predict(self, inp, inv_inp):
        return self.session.run(self.Qsa, feed_dict={self.state_inp: inp, self.inventory: inv_inp})