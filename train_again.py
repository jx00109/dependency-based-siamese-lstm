# -*- coding:utf8 -*-
import tensorflow as tf
import numpy as np
import os
import time
from lstmRNN import LSTMRNN
import dataHelperMask
from scipy.stats import pearsonr, spearmanr
from gensim.models.word2vec import KeyedVectors

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 32, 'the batch_size of the training procedure')
flags.DEFINE_float('lr', 0.0001, 'the learning rate')
flags.DEFINE_integer('embedding_dim', 300, 'embedding dim')
flags.DEFINE_integer('hidden_neural_size', 50, 'LSTM hidden neural size')
flags.DEFINE_integer('max_len', 40, 'max_len of training sentence')
flags.DEFINE_float('alpha_weight', 0.4, 'weight of sentence subj and obj')  # 注意修改此处
flags.DEFINE_integer('pre_num_epoch', 1501, 'num epoch')
flags.DEFINE_integer('num_epoch', 5301, 'num epoch')
flags.DEFINE_string('out_dir', os.path.abspath(os.path.join(os.path.curdir, "results/alpha-0.4-nopre")),
                    'output directory')  # 注意修改此处
flags.DEFINE_string('pre_logs_dir', './logs/pre-model-0.4.txt', 'pre train logs')  # 注意修改此处
flags.DEFINE_string('logs_dir', './logs/model-0.4-nopre.txt', 'train logs')  # 注意修改此处
flags.DEFINE_integer('check_point_every', 100, 'checkpoint every num epoch ')
flags.DEFINE_integer('pre_check_point_every', 250, 'pre checkpoint every num epoch ')
flags.DEFINE_string('checkpoint2train', './results/alpha-0.5/checkpoints/model-4011660', 'contine2train model dir')


class Config(object):
    hidden_neural_size = FLAGS.hidden_neural_size
    embed_dim = FLAGS.embedding_dim
    lr = FLAGS.lr
    batch_size = FLAGS.batch_size
    num_step = FLAGS.max_len
    num_epoch = FLAGS.num_epoch
    out_dir = FLAGS.out_dir
    checkpoint_every = FLAGS.check_point_every


def cut_data(data, rate):
    x1, x2, y, mask_x1, mask_x2 = data

    n_samples = len(x1)

    # 打散数据集
    sidx = np.random.permutation(n_samples)

    ntrain = int(np.round(n_samples * (1.0 - rate)))

    train_x1 = [x1[s] for s in sidx[:ntrain]]
    train_x2 = [x2[s] for s in sidx[:ntrain]]
    train_y = [y[s] for s in sidx[:ntrain]]
    train_m1 = [mask_x1[s] for s in sidx[:ntrain]]
    train_m2 = [mask_x2[s] for s in sidx[:ntrain]]

    valid_x1 = [x1[s] for s in sidx[ntrain:]]
    valid_x2 = [x2[s] for s in sidx[ntrain:]]
    valid_y = [y[s] for s in sidx[ntrain:]]
    valid_m1 = [mask_x1[s] for s in sidx[ntrain:]]
    valid_m2 = [mask_x2[s] for s in sidx[ntrain:]]

    # 打散划分好的训练和测试集
    train_data = [train_x1, train_x2, train_y, train_m1, train_m2]
    valid_data = [valid_x1, valid_x2, valid_y, valid_m1, valid_m2]

    return train_data, valid_data


def evaluate(model, session, data, istest=False):
    x1, x2, y, mask_x1, mask_x2 = data

    fetches = [model.truecost, model.sim, model.target]
    feed_dict = {}
    feed_dict[model.input_data_s1] = x1
    feed_dict[model.input_data_s2] = x2
    feed_dict[model.target] = y
    feed_dict[model.mask_s1] = mask_x1
    feed_dict[model.mask_s2] = mask_x2
    model.assign_new_batch_size(session, len(x1))
    cost, sim, target = session.run(fetches, feed_dict)
    if istest:
        pearson_r, _ = pearsonr(sim, target)
        spearman_r, _ = spearmanr(sim, target)
        return cost, pearson_r, spearman_r

    return cost


def run_epoch(model, session, data, global_steps, valid_model, valid_data):
    for step, (s1, s2, y, mask_s1, mask_s2) in enumerate(
            dataHelperMask.batch_iter(data, batch_size=FLAGS.batch_size)):

        feed_dict = {}
        feed_dict[model.input_data_s1] = s1
        feed_dict[model.input_data_s2] = s2
        feed_dict[model.target] = y
        feed_dict[model.mask_s1] = mask_s1
        feed_dict[model.mask_s2] = mask_s2
        model.assign_new_batch_size(session, len(s1))
        fetches = [model.truecost, model.sim, model.target, model.train_op]
        cost, sim, target, _ = session.run(fetches, feed_dict)

        if (global_steps % 100 == 0):
            valid_cost = evaluate(valid_model, session, valid_data, istest=False)
            print "step %i , train cost: %f  valid cost: %f " % (global_steps, cost, valid_cost)

        global_steps += 1

    return global_steps


def train_step():
    config = Config()

    with tf.Graph().as_default(), tf.Session() as session:
        # initializer = tf.random_normal_initializer(0.0, 0.02, dtype=tf.float64)
        with tf.variable_scope("mymodel", reuse=None, dtype=tf.float64):
            model = LSTMRNN(config=config, is_training=True)

        with tf.variable_scope("mymodel", reuse=True, dtype=tf.float64):
            valid_model = LSTMRNN(config=config, is_training=False)
            test_model = LSTMRNN(config=config, is_training=False)

        # add checkpoint
        pre_checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints-pre"))
        pre_checkpoint_prefix = os.path.join(pre_checkpoint_dir, "model-pre")
        if not os.path.exists(pre_checkpoint_dir):
            os.makedirs(pre_checkpoint_dir)

        checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(session, FLAGS.checkpoint2train)
        # tf.global_variables_initializer().run()

        global_steps = 1984156
        index = np.random.randint(1, 4000)
        begin_time = int(time.time())
        print("loading the dataset...")

        pretrained_word_model = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)
        '''
        pre_train_data = dataHelperMask.load_data(FLAGS.max_len, pretrained_word_model,
                                                  datapath='./data/stsallrmf.mask2.p',
                                                  embed_dim=FLAGS.embedding_dim, alpha=FLAGS.alpha_weight)
        print "index: ", index

        print "begin pre-training"

        for i in range(FLAGS.pre_num_epoch):
            print "the %d epoch pre-training..." % (i + 1)

            lr = model.assign_new_lr(session, config.lr)

            print "current learning rate is %f" % lr
            # 11000+数据
            train_data, valid_data = cut_data(pre_train_data, 0.005)
            global_steps = run_epoch(model, session, train_data, global_steps, valid_model, valid_data)

            if i % FLAGS.pre_check_point_every == 0:
                path = saver.save(session, pre_checkpoint_prefix, global_steps)
                pos = 'index: %d, position: %s' % (i, path)
                with open(FLAGS.pre_logs_dir, 'a') as f:
                    f.write(pos)
                    f.write('\n')
                print pos

        print "pre-train finish."
        '''
        print "begin training"
        #data = dataHelperMask.load_data(FLAGS.max_len, pretrained_word_model, datapath='./data/semtrain.mask2.p',
        #                                embed_dim=FLAGS.embedding_dim, alpha=FLAGS.alpha_weight)
        test_data = dataHelperMask.load_data(FLAGS.max_len, pretrained_word_model, datapath='./data/semtest.mask2.p',
                                             embed_dim=FLAGS.embedding_dim, alpha=FLAGS.alpha_weight)

        # print "length of train set:", len(data[2])
        # print "example of train set: ", data[3][index]
        print "length of test set:", len(test_data[2])
        print "example of test set:", test_data[3][index]

	x1, x2, y, m1, m2 = test_data
	fetches=[test_model.sent1, test_model.sent2]
	feed_dict={}
	feed_dict[test_model.input_data_s1] = x1
	feed_dict[test_model.input_data_s2] = x2
	feed_dict[test_model.target] = y
	feed_dict[test_model.mask_s1] = m1
	feed_dict[test_model.mask_s2] = m2
	test_model.assign_new_batch_size(session, len(x1))
	sentence1, sentence2 = session.run(fetches, feed_dict)
	
	print 'type of sent1: ', type(sentence1)
	print 'len of sent1: ', len(sentence1)
	print 'type of sent2: ', type(sentence2)
        print 'len of sent2: ', len(sentence2)

        '''
        for i in range(config.num_epoch):
            print("the %d epoch training..." % (i + 1))

            lr = model.assign_new_lr(session, config.lr)

            print "current learning rate is %f" % lr

            train_data, valid_data = cut_data(data, 0.01)

            global_steps = run_epoch(model, session, train_data, global_steps, valid_model, valid_data)

            if i % config.checkpoint_every == 0 and i != 0:
                path = saver.save(session, checkpoint_prefix, global_steps)
                test_cost, test_pearson_r, test_spearman_r = evaluate(test_model, session, test_data, istest=True)
                res = 'index: %d, cost: %f, pearson_r: %f, spearman_r: %f' % (
                    i, test_cost, test_pearson_r, test_spearman_r)
                print res
                with open(FLAGS.logs_dir, 'a') as f:
                    f.write(res)
                    f.write('\n')
                    f.write('model position: {}\n'.format(path))
                print("Saved results chechpoint to{}\n".format(path))

        print("the train is finished")
        end_time = int(time.time())
        print("training takes %d seconds already\n" % (end_time - begin_time))
        '''
        print("program end!")


def main(_):
    train_step()


if __name__ == "__main__":
    tf.app.run()
