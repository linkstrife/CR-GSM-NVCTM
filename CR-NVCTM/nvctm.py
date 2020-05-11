# -*- coding: utf-8 -*-
"""NVCTM Tensorflow implementation by Jiachun Feng"""
from __future__ import print_function

import numpy as np
import tensorflow as tf
import npmi
import os
import utils as utils
import pickle
import codecs

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_options = tf.GPUOptions(allow_growth=True)
tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

flags = tf.app.flags
flags.DEFINE_string('data_dir', './data/StackOverflow', 'Data dir path.')
flags.DEFINE_string('model_dir', './data/', 'Model dir path.')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('n_hidden', 256, 'Size of each hidden layer.')
flags.DEFINE_integer('n_topic', 50, 'Size of stochastic vector.')
flags.DEFINE_integer('n_sample', 20, 'Number of samples.')
flags.DEFINE_integer('vocab_size', 22956, 'Vocabulary size.')  # StackOverflow: 22956 Snippets: 30642
flags.DEFINE_boolean('test', True, 'Process test data.')
flags.DEFINE_string('non_linearity', 'tanh', 'Non-linearity of the MLP.')
flags.DEFINE_integer('n_householder', 10, 'Number of Householder transformer.')
FLAGS = flags.FLAGS


class NVCTM(object):
    def __init__(self,
                 vocab_size,
                 n_hidden,
                 n_topic,
                 n_sample,
                 learning_rate,
                 batch_size,
                 n_householder,
                 non_linearity):
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self.n_sample = n_sample
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_householder = n_householder

        self.x = tf.placeholder(tf.float32, [None, vocab_size], name='input')
        self.mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings

        # encoder
        with tf.variable_scope('encoder'):
            self.enc_vec = utils.mlp(self.x, [self.n_hidden], self.non_linearity)
            self.mean = utils.linear(self.enc_vec, self.n_topic, scope='mean')
            self.logsigm = utils.linear(self.enc_vec,
                                        self.n_topic,
                                        bias_start_zero=True,
                                        matrix_start_zero=True,
                                        scope='logsigm')

            # -------------------- cal the householder matrix -------------------------------
            self.tmp_mean = tf.expand_dims(
                tf.expand_dims(tf.rsqrt(tf.reduce_sum(tf.square(self.mean), 1)), 1) * self.mean, 2)
            self.tmp_mean_t = tf.transpose(self.tmp_mean, perm=[0, 2, 1])
            self.vk = self.tmp_mean
            self.Hk = tf.expand_dims(tf.eye(self.n_topic), 0) - \
                      2 * tf.matmul(self.tmp_mean, self.tmp_mean_t)

            self.U = self.Hk
            self.tmp_vk = self.vk
            self.invalid = []
            self.vk_show = tf.constant(-1.0)
            for k in range(1, self.n_householder + 1):
                self.tmp_vk = self.vk
                self.tmp_vk = tf.expand_dims(
                    tf.rsqrt(tf.reduce_sum(tf.square(self.tmp_vk), 1)) * tf.squeeze(self.tmp_vk, [2, 2]), 2)
                self.vk = tf.matmul(self.Hk, self.vk)
                self.Hk = tf.expand_dims(tf.eye(self.n_topic), 0) - \
                          2 * tf.matmul(self.tmp_vk, tf.transpose(self.tmp_vk, perm=[0, 2, 1]))
                self.U = tf.matmul(self.U, self.Hk)

            self.Umean = tf.squeeze(tf.matmul(self.U, self.tmp_mean), [2, 2])

            # ------------------------ KL divergence after Householder -------------------------------------
            self.kld = -0.5 * (tf.reduce_sum(
                1 - tf.square(self.Umean) + 2 * self.logsigm, 1) - \
                               tf.trace(tf.matmul(tf.transpose(tf.multiply(tf.expand_dims(tf.exp(2 * self.logsigm), 2),
                                                                           tf.transpose(self.U, perm=[0, 2, 1])),
                                                               perm=[0, 2, 1]), tf.transpose(self.U, perm=[0, 2, 1]))))
            # kk = tf.trace(tf.matmul(tf.transpose(tf.multiply(tf.expand_dims(tf.exp(2 * self.logsigm), 2), tf.transpose(self.U, perm=[0,2,1])), perm=[0,2,1]), tf.transpose(self.U, perm=[0,2,1])))
            self.log_squre = tf.trace(tf.matmul(tf.transpose(
                tf.multiply(tf.expand_dims(tf.exp(2 * self.logsigm), 2), tf.transpose(self.U, perm=[0, 2, 1])),
                perm=[0, 2, 1]), tf.transpose(self.U, perm=[0, 2, 1])))
            self.mean_squre = tf.reduce_sum(tf.square(self.Umean), 1)
            self.kld = self.mask * self.kld  # mask paddings

            if self.n_sample == 1:  # single sample
                eps = tf.random_normal((batch_size, self.n_topic), 0, 1)
                doc_vec = tf.multiply(tf.exp(self.logsigm), eps) + self.mean
            else:
                doc_vec_list = []
                for i in range(self.n_sample):
                    epsilon = tf.random_normal((self.batch_size, self.n_topic), 0, 1)
                    doc_vec_list.append(self.mean + tf.multiply(epsilon, tf.exp(self.logsigm)))
                doc_vec = tf.add_n(doc_vec_list) / self.n_sample

            doc_vec = tf.squeeze(tf.matmul(self.U, tf.expand_dims(doc_vec, 2)))
            self.theta = tf.nn.softmax(doc_vec)

        with tf.variable_scope('decoder'):
            topic_vec = tf.get_variable('topic_vec', shape=[self.n_topic, self.n_hidden])
            word_vec = tf.get_variable('word_vec', shape=[self.vocab_size, self.n_hidden])

            # n_topic x vocab_size
            beta = tf.matmul(topic_vec, tf.transpose(word_vec))

            logits = tf.nn.log_softmax(tf.matmul(doc_vec, beta))

            self.beta = tf.nn.softmax(beta)

            mean = tf.reduce_mean(self.theta, -1, keep_dims=True)  # bs x 1
            self.variance = tf.sqrt(
                tf.reduce_sum(tf.square(self.theta - tf.tile(mean, [1, self.n_topic])), -1) / self.n_topic)

            self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x), 1)

        self.objective = self.recons_loss + self.kld
        self.loss_func = self.objective

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        fullvars = tf.trainable_variables()

        enc_vars = utils.variable_parser(fullvars, 'encoder')
        dec_vars = utils.variable_parser(fullvars, 'decoder')

        enc_grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_func, enc_vars), 5)
        dec_grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_func, dec_vars), 5)

        self.optim_enc = optimizer.apply_gradients(zip(enc_grads, enc_vars))
        self.optim_dec = optimizer.apply_gradients(zip(dec_grads, dec_vars))


def train(sess, model,
          train_url,
          test_url,
          dev_url,
          model_url,
          batch_size,
          saver,
          training_epochs=400,
          alternate_epochs=1):
    """train nvctm model."""
    train_set, train_count = utils.data_set(train_url)
    dev_set, dev_count = utils.data_set(dev_url)
    test_set, test_count = utils.data_set(test_url)

    dev_batches = utils.create_batches(len(dev_set), batch_size, shuffle=False)
    test_batches = utils.create_batches(len(test_set), batch_size, shuffle=False)

    train_theta = []
    train_beta = []
    for epoch in range(training_epochs):
        train_batches = utils.create_batches(len(train_set), batch_size, shuffle=True)
        # -------------------------------
        # train
        for switch in range(0, 2):
            if switch == 0:
                optim = model.optim_dec
                print_mode = 'updating decoder'
            else:
                optim = model.optim_enc
                print_mode = 'updating encoder'
            for i in range(alternate_epochs):
                loss_sum = 0.0
                ppx_sum = 0.0
                kld_sum = 0.0
                word_count = 0
                doc_count = 0
                res_sum = 0
                log_sum = 0
                mean_sum = 0
                var_sum = 0
                m = None
                Um = None
                enc = None

                for idx_batch in train_batches:
                    data_batch, count_batch, mask = utils.fetch_data(
                        train_set, train_count, idx_batch, FLAGS.vocab_size)
                    input_feed = {model.x.name: data_batch, model.mask.name: mask}
                    _, (loss, kld, mean, Umean, enc, rec_loss, log_s, mean_s, vk_show, theta, beta, lp, v) = sess.run(
                        (optim, [model.objective, model.kld, model.mean, model.U, model.vk, model.recons_loss,
                          model.log_squre, model.mean_squre, model.vk_show, model.theta, model.beta,
                          model.log_prob, model.variance]), input_feed)
                    m = mean
                    Um = Umean
                    # print('*********************vk show', vk_show)
                    # print('Umean', Umean[0])
                    loss_sum += np.sum(loss)
                    kld_sum += np.sum(kld) / np.sum(mask)
                    word_count += np.sum(count_batch)
                    res_sum += np.sum(rec_loss)
                    log_sum += np.sum(log_s)
                    mean_sum += np.sum(mean_s)
                    var_sum += np.sum(v) / np.sum(mask)
                    # to avoid nan error
                    count_batch = np.add(count_batch, 1e-12)
                    # per document loss
                    ppx_sum += np.sum(np.divide(loss, count_batch))
                    doc_count += np.sum(mask)

                    if epoch == training_epochs - 1 and switch == 1 and i == alternate_epochs - 1:
                        train_theta.extend(theta)
                        train_beta.extend(beta)

                print_ppx = np.exp(loss_sum / word_count)
                # print_ppx_perdoc = np.exp(ppx_sum / doc_count)
                print_kld = kld_sum / len(train_batches)
                print_res = res_sum / len(train_batches)
                print_log = log_sum / len(train_batches)
                print_mean = mean_sum / len(train_batches)
                print_var = var_sum / len(train_batches)

                print('| Epoch train: {:d} |'.format(epoch + 1),
                      print_mode, '{:d}'.format(i),
                      '| Corpus ppx: {:.5f}'.format(print_ppx),  # perplexity per word
                      # '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),  # perplexity for per doc
                      '| KLD: {:.5}'.format(print_kld),
                      '| stddev {:.5}'.format(print_var),
                      '| res_loss: {:5}'.format(print_res),
                      '| log_loss: {:5}'.format(print_log),
                      '| mean_loss: {:5}'.format(print_mean))

                with codecs.open('./nvctm_train_theta', 'wb') as fp:
                    pickle.dump(np.array(train_theta), fp)
                fp.close()

                if (epoch + 1) % 50 == 0 and switch == 1 and i == alternate_epochs - 1:
                    with codecs.open('./nvctm_train_beta', 'wb') as fp:
                        pickle.dump(beta, fp)
                    fp.close()
                    npmi.print_coherence('nvctm', FLAGS.data_dir + '/train.feat', FLAGS.vocab_size)

        # dev
        loss_sum = 0.0
        kld_sum = 0.0
        ppx_sum = 0.0
        var_sum = 0
        word_count = 0
        doc_count = 0
        for idx_batch in dev_batches:
            data_batch, count_batch, mask = utils.fetch_data(
                dev_set, dev_count, idx_batch, FLAGS.vocab_size)
            input_feed = {model.x.name: data_batch, model.mask.name: mask}
            loss, kld, v = sess.run([model.objective, model.kld, model.variance], input_feed)
            loss_sum += np.sum(loss)
            kld_sum += np.sum(kld) / np.sum(mask)
            var_sum += np.sum(v) / np.sum(mask)
            word_count += np.sum(count_batch)
            count_batch = np.add(count_batch, 1e-12)
            ppx_sum += np.sum(np.divide(loss, count_batch))
            doc_count += np.sum(mask)
        print_ppx = np.exp(loss_sum / word_count)
        print_var = var_sum / len(train_batches)
        # print_ppx_perdoc = np.exp(ppx_sum / doc_count)
        print_kld = kld_sum / len(dev_batches)
        print('\n| Epoch dev: {:d}'.format(epoch + 1),
              '| Perplexity: {:.9f}'.format(print_ppx),
              '| stddev {:.5}'.format(print_var),
              '| KLD: {:.5}'.format(print_kld))

        # test
        if FLAGS.test:
            loss_sum = 0.0
            kld_sum = 0.0
            ppx_sum = 0.0
            var_sum = 0.0
            word_count = 0
            doc_count = 0
            for idx_batch in test_batches:
                data_batch, count_batch, mask = utils.fetch_data(
                    test_set, test_count, idx_batch, FLAGS.vocab_size)
                input_feed = {model.x.name: data_batch, model.mask.name: mask}
                loss, kld, v = sess.run([model.objective, model.kld, model.variance], input_feed)
                loss_sum += np.sum(loss)
                kld_sum += np.sum(kld) / np.sum(mask)
                var_sum += np.sum(v) / np.sum(mask)
                word_count += np.sum(count_batch)
                count_batch = np.add(count_batch, 1e-12)
                ppx_sum += np.sum(np.divide(loss, count_batch))
                doc_count += np.sum(mask)
            print_ppx = np.exp(loss_sum / word_count)
            print_var = var_sum / len(train_batches)
            # print_ppx_perdoc = np.exp(ppx_sum / doc_count)
            print_kld = kld_sum / len(test_batches)
            print('| Epoch test: {:d}'.format(epoch + 1),
                  '| Perplexity: {:.9f}'.format(print_ppx),
                  '| stddev {:.5}'.format(print_var),
                  '| KLD: {:.5}\n'.format(print_kld))
    npmi.print_coherence('nvctm', FLAGS.data_dir + '/train.feat', FLAGS.vocab_size)
    saver.save(sess, model_url)


def main(argv=None):
    if FLAGS.non_linearity == 'tanh':
        non_linearity = tf.nn.tanh
    elif FLAGS.non_linearity == 'sigmoid':
        non_linearity = tf.nn.sigmoid
    else:
        non_linearity = tf.nn.relu

    nvctm = NVCTM(vocab_size=FLAGS.vocab_size,
                  n_hidden=FLAGS.n_hidden,
                  n_topic=FLAGS.n_topic,
                  n_sample=FLAGS.n_sample,
                  learning_rate=FLAGS.learning_rate,
                  batch_size=FLAGS.batch_size,
                  n_householder=FLAGS.n_householder,
                  non_linearity=non_linearity)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    train_url = os.path.join(FLAGS.data_dir, 'train.feat')
    test_url = os.path.join(FLAGS.data_dir, 'test.feat')
    dev_url = os.path.join(FLAGS.data_dir, 'dev.feat')
    saver = tf.train.Saver()
    model_url = os.path.join(FLAGS.model_dir, '')

    train(sess=sess,
          model=nvctm,
          train_url=train_url,
          test_url=test_url,
          dev_url=dev_url,
          model_url=model_url,
          batch_size=FLAGS.batch_size,
          saver=saver)


if __name__ == '__main__':
    tf.app.run()
