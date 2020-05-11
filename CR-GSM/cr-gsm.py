# -*- coding: utf-8 -*-
# @Time : 2019/8/15 上午9:35
# @Author : Lihui Lin
# @FileName: cr-gsm.py

import tensorflow as tf
import npmi
import numpy as np
import utils
import codecs
import os
import pickle

flags = tf.flags
flags.DEFINE_string('data_dir', './data/StackOverflow', 'The directory of training data.')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate for the model.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('n_hidden', 256, 'Number of hidden nodes.')
flags.DEFINE_integer('n_topic', 50, 'Size of the stochastic topic vector.')
flags.DEFINE_integer('n_sample', 20, 'Number of samples.')
flags.DEFINE_integer('vocab_size', 22956, 'Vocabulary size.')  # StackOverflow: 22956 Snippets: 30642
flags.DEFINE_boolean('test', True, 'Process test data.')
flags.DEFINE_string('non_linearity', 'tanh', 'Non-linearity of the MLP.')
flags.DEFINE_string('model_type', 'topic', 'Switch between topic and document model.')
FLAGS = flags.FLAGS


class GSM(object):
    def __init__(self, vocab_size, n_hidden, n_topic, n_sample, learning_rate, batch_size, non_linearity, model_type):
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self. n_sample = n_sample
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.non_linearity = non_linearity
        self.model_type = model_type
        self.is_training = tf.placeholder(tf.bool, name='training_flag')
        self.gamma = tf.placeholder(tf.float32, name='weight')

        # batch_size x vocab_size
        self.x = tf.placeholder(tf.float32, [batch_size, vocab_size], name='input')
        # 用于序列补0
        self.mask = tf.placeholder(tf.float32, [batch_size], name='mask')

        with tf.variable_scope('Encoder'):
            # feed to VAE
            self.enc_vec = tf.layers.dense(self.x, self.n_hidden, tf.nn.tanh)  # encode document embedding
            self.mean = tf.layers.dense(self.enc_vec, self.n_topic)  # 均值模块，dim_doc -> dim_topic
            self.log_sigma = tf.layers.dense(self.enc_vec, self.n_topic)  # 方差模块

            # KL(Norm(sigma, miu^2)||Norm(miu0, sigma0^2))
            self.kld = -0.5 * tf.reduce_sum(1 - tf.square(self.mean) + 2 * self.log_sigma - tf.exp(2 * self.log_sigma), 1)
            self.kld = tf.multiply(self.mask, self.kld)

            doc_vec_list = []
            for i in range(self.n_sample):
                epsilon = tf.random_normal((self.batch_size, self.n_topic), 0, 1)
                doc_vec_list.append(self.mean + tf.multiply(epsilon, tf.exp(self.log_sigma)))
            doc_vec = tf.add_n(doc_vec_list)/self.n_sample

        with tf.variable_scope('Decoder'):
            # self.log_lambd = tf.layers.dense(self.enc_vec, 1)
            # self.lambd = tf.exp(self.log_lambd) + 1e-5
            self.lambd = tf.constant(shape=[self.batch_size, 1], value=1.)

            # batch_size x self.n_topic
            if self.model_type == 'topic':
                self.m_theta = tf.layers.dense(doc_vec, self.n_topic)
                self.theta = tf.nn.softmax(self.m_theta)
                self.topic_dist = self.theta

                mean = tf.reduce_mean(self.theta, -1, keep_dims=True)  # bs x 1, 1/n_topic
                self.variance = tf.sqrt(tf.reduce_sum(tf.square(self.theta - tf.tile(mean, [1, self.n_topic])), -1)/self.n_topic)
                self.log_prob = (-self.n_topic-(1./self.lambd)) * tf.log(
                    tf.reduce_sum(tf.pow(self.theta, -self.lambd), -1, keep_dims=True) - self.n_topic + 1)
                # this term can be omitted whe lambda is a constant parameter
                # might be very large when the topic number is big
                constant_term = 0.0
                for i in range(self.n_topic):
                    constant_term += tf.log(1 + i*self.lambd)
                self.log_prob += constant_term

                # tune this parameter C to balance gradients and stablize training
                self.log_prob += 300
                self.log_prob = tf.clip_by_value(self.log_prob, 0, np.inf)

            elif self.model_type == 'document':
                self.theta = doc_vec  # remove the softmax

            topic_vec = tf.get_variable('topic_vec', shape=[self.n_topic, self.n_hidden])
            word_vec  = tf.get_variable('word_vec',  shape=[self.vocab_size, self.n_hidden])

            # n_topic x vocab_size
            self.beta = tf.nn.softmax(tf.matmul(topic_vec, tf.transpose(word_vec)))

            if self.model_type == 'topic':
                self.d_given_theta = tf.log(tf.matmul(self.theta, self.beta))
            elif self.model_type == 'document':
                self.d_given_theta = tf.nn.log_softmax(tf.matmul(self.m_theta, self.beta))

            self.reconstruction_loss = -tf.reduce_sum(tf.multiply(self.d_given_theta, self.x), 1)

        self.objective = self.reconstruction_loss + self.kld
        self.loss_func = self.objective + 0.05*self.log_prob

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        full_vars = tf.trainable_variables()
        variable_names = [v.name for v in full_vars]
        print(variable_names)

        enc_var = utils.variable_parser(full_vars, 'Encoder')
        dec_var = utils.variable_parser(full_vars, 'Decoder')

        enc_grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss_func, enc_var), 5)
        dec_grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss_func, dec_var), 5)

        self.optimize_enc = optimizer.apply_gradients(zip(enc_grad, enc_var))
        self.optimize_dec = optimizer.apply_gradients(zip(dec_grad, dec_var))


def train(sess, model, train_url, test_url, dev_url, batch_size, training_epochs=1000, alternate_epochs=1):
    """train cr-gsm model."""
    train_set, train_count = utils.data_set(train_url)
    test_set, test_count = utils.data_set(test_url)
    dev_set, dev_count = utils.data_set(dev_url)

    dev_batches = utils.create_batches(len(dev_set), batch_size, shuffle=False)
    test_batches = utils.create_batches(len(test_set), batch_size, shuffle=False)

    kld_list = []
    var_list = []
    train_theta = []
    train_beta = []
    test_theta = []
    test_beta = []
    for epoch in range(training_epochs):
        train_batches = utils.create_batches(len(train_set), batch_size, shuffle=True)
        # -------------------------------
        # train
        for switch in range(0, 2):
            if switch == 0:
                optimize = model.optimize_dec
                print_mode = 'updating decoder'
            elif switch == 1:
                optimize = model.optimize_enc
                print_mode = 'updating encoder'
            for i in range(alternate_epochs):
                loss_sum = 0.0
                ppx_sum = 0.0
                kld_sum = 0.0
                word_count = 0
                doc_count = 0
                var_sum = 0
                for idx_batch in train_batches:
                    data_batch, count_batch, mask = utils.fetch_data(
                        train_set, train_count, idx_batch, FLAGS.vocab_size)

                    input_feed = {model.x.name: data_batch, model.mask.name: mask, model.is_training.name: True, model.gamma.name: epoch/training_epochs}
                    _, (loss, kld, v, theta, beta, lp) =\
                        sess.run((optimize, [model.reconstruction_loss, model.kld, model.variance, model.topic_dist,
                                             model.beta, model.log_prob]), input_feed)
                    loss_sum += np.sum(loss)
                    kld_sum += np.sum(kld) / np.sum(mask)
                    var_sum += np.sum(v) / np.sum(mask)
                    # print([np.max(theta[i]) for i in range(batch_size)])
                    # print([np.argmax(theta[i]) for i in range(batch_size)])
                    word_count += np.sum(count_batch)
                    # to avoid nan error
                    count_batch = np.add(count_batch, 1e-12)
                    # per document loss
                    ppx_sum += np.sum(np.divide(loss, count_batch))
                    doc_count += np.sum(mask)

                    if epoch == training_epochs - 1 and switch == 1 and i == alternate_epochs - 1:
                        train_theta.extend(theta)

                print_ppx = np.exp(loss_sum / word_count)
                print_ppx_perdoc = np.exp(ppx_sum / doc_count)
                print_kld = kld_sum / len(train_batches)
                print_var = var_sum / len(train_batches)
                kld_list.append(print_kld)
                var_list.append(print_var)
                print('| Epoch train: {:d}'.format(epoch + 1),
                      print_mode, '{:d}'.format(i + 1),
                      '| Corpus ppx: {:.5f}'.format(print_ppx),  # perplexity for all docs
                      '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),  # perplexity for per doc
                      '| KLD: {:.5}'.format(print_kld),
                      '| stddev {:.5}'.format(print_var))

                with codecs.open('./cr_gsm_train_theta', 'wb') as fp:
                    pickle.dump(np.array(train_theta), fp)
                fp.close()

                if (epoch + 1) % 50 == 0 and switch == 1 and i == alternate_epochs - 1:
                    with codecs.open('./cr_gsm_train_beta', 'wb') as fp:
                        pickle.dump(beta, fp)
                    fp.close()
                    npmi.print_coherence('cr_gsm', FLAGS.data_dir + '/train.feat', FLAGS.vocab_size)

        # -------------------------------
        # dev
        loss_sum = 0.0
        kld_sum = 0.0
        ppx_sum = 0.0
        word_count = 0
        doc_count = 0
        var_sum = 0
        for idx_batch in dev_batches:
            data_batch, count_batch, mask = utils.fetch_data(dev_set, dev_count, idx_batch, FLAGS.vocab_size)
            input_feed = {model.x.name: data_batch, model.mask.name: mask, model.is_training.name: False, model.gamma.name: 0}
            loss, kld, v = sess.run([model.objective, model.kld, model.variance], input_feed)
            loss_sum += np.sum(loss)
            kld_sum += np.sum(kld) / np.sum(mask)
            word_count += np.sum(count_batch)
            count_batch = np.add(count_batch, 1e-12)
            ppx_sum += np.sum(np.divide(loss, count_batch))
            var_sum += np.sum(v) / np.sum(mask)
            doc_count += np.sum(mask)
        print_ppx = np.exp(loss_sum / word_count)
        print_ppx_perdoc = np.exp(ppx_sum / doc_count)
        print_kld = kld_sum / len(dev_batches)
        print_var = var_sum / len(train_batches)
        print('\n| Epoch dev: {:d}'.format(epoch + 1),
              '| Perplexity: {:.9f}'.format(print_ppx),
              '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
              '| KLD: {:.5}'.format(print_kld),
              '| stddev: {:.5}'.format(print_var))

        # test
        if FLAGS.test:
            loss_sum = 0.0
            kld_sum = 0.0
            ppx_sum = 0.0
            word_count = 0
            doc_count = 0
            for idx, idx_batch in enumerate(test_batches):
                data_batch, count_batch, mask = utils.fetch_data(
                    test_set, test_count, idx_batch, FLAGS.vocab_size)
                input_feed = {model.x.name: data_batch, model.mask.name: mask, model.is_training.name: False, model.gamma.name: 0}
                loss, kld, theta, beta, v = sess.run([model.objective, model.kld, model.topic_dist, model.beta, model.variance], input_feed)
                loss_sum += np.sum(loss)
                kld_sum += np.sum(kld) / np.sum(mask)
                word_count += np.sum(count_batch)
                count_batch = np.add(count_batch, 1e-12)
                ppx_sum += np.sum(np.divide(loss, count_batch))
                doc_count += np.sum(mask)
                test_theta.extend(theta)
                if idx == len(test_batches) - 1:
                    test_beta.extend(beta)
            print_ppx = np.exp(loss_sum / word_count)
            print_ppx_perdoc = np.exp(ppx_sum / doc_count)
            print_kld = kld_sum / len(test_batches)
            print('| Epoch test: {:d}'.format(epoch + 1),
                  '| Perplexity: {:.9f}'.format(print_ppx),
                  '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
                  '| KLD: {:.5}'.format(print_kld),
                  '| stddev: {:.5}\n'.format(print_var))

    with codecs.open('./test_theta', 'wb') as fp:
        pickle.dump(test_theta, fp)
    fp.close()

    with codecs.open('./test_beta', 'wb') as fp:
        pickle.dump(test_beta, fp)
    fp.close()

    with codecs.open('./kld.txt', 'w', 'utf-8') as fp:
        for idx, kld in enumerate(kld_list):
            if idx < len(kld_list) - 1:
                fp.write(str(kld) + ', ')
            else:
                fp.write(str(kld))
        fp.close()
    with codecs.open('./var.txt', 'w', 'utf-8') as fp:
        for idx, var in enumerate(var_list):
            if idx < len(var_list) - 1:
                fp.write(str(var) + ', ')
            else:
                fp.write(str(var))
        fp.close()


def main(argv=None):
    if FLAGS.non_linearity == 'tanh':
        non_linearity = tf.nn.tanh
    elif FLAGS.non_linearity == 'sigmoid':
        non_linearity = tf.nn.sigmoid
    else:
        non_linearity = tf.nn.leaky_relu

    gsm = GSM(vocab_size=FLAGS.vocab_size,
              n_hidden=FLAGS.n_hidden,
              n_topic=FLAGS.n_topic,
              n_sample=FLAGS.n_sample,
              learning_rate=FLAGS.learning_rate,
              batch_size=FLAGS.batch_size,
              non_linearity=non_linearity,
              model_type=FLAGS.model_type)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    train_url = os.path.join(FLAGS.data_dir, 'train.feat')
    test_url = os.path.join(FLAGS.data_dir, 'test.feat')
    dev_url = os.path.join(FLAGS.data_dir, 'dev.feat')

    train(sess, gsm, train_url, test_url, dev_url, FLAGS.batch_size)


if __name__ == '__main__':
    tf.app.run()
