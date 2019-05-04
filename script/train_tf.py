import argparse
import time
import json
import numpy as np
import math
import random
import tensorflow as tf

def batch_generator(X, y, batch_size=128, return_idx=False, crf=False):
    for offset in range(0, X.shape[0], batch_size):
        batch_X_len = np.sum(X[offset:offset + batch_size] != 0, axis=1)
        batch_idx = batch_X_len.argsort()[::-1]
        batch_X_len = batch_X_len[batch_idx]
        batch_X_mask = (X[offset:offset + batch_size] != 0)[batch_idx].astype(np.uint8)
        batch_X = X[offset:offset + batch_size][batch_idx]
        batch_y = y[offset:offset + batch_size][batch_idx]

        if return_idx:  # in testing, need to sort back.
            yield (batch_X, batch_y, batch_X_len, batch_X_mask, batch_idx)
        else:
            yield (batch_X, batch_y, batch_X_len, batch_X_mask)

class Model(object):
    def __init__(self, gen_emb, domain_emb, num_classes=3, crf=False, testing=False):

        max_sentence_size = 83
        self.x = tf.placeholder(tf.int32, shape=[None, max_sentence_size])
        self.x_len = tf.placeholder(tf.int32, shape=[None, ])
        self.input_y = tf.placeholder(tf.int32, [None, max_sentence_size])
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        # Embedding
        gen_embedding = tf.Variable(gen_emb, name='gen_embedding', dtype=tf.float32, trainable=False)
        gen_emb_in = tf.nn.embedding_lookup(gen_embedding, self.x)
        domain_embedding = tf.Variable(domain_emb, name='domain_embedding', dtype=tf.float32, trainable=False)
        domain_emb_in = tf.nn.embedding_lookup(domain_embedding, self.x)
        x_emb = tf.concat([gen_emb_in, domain_emb_in], axis=-1)
        x_emb = tf.nn.dropout(x_emb, self.dropout)

        # Conv
        # with tf.variable_scope("x_conv_0"):
        #     x_conv_0 = tf.layers.conv1d(x_emb, 128, kernel_size=3, dilation_rate=2, padding='same', activation=None)
        x_conv_1 = self.conv(x_emb, 128, activation=None, kernel_size=5, name="x_conv_1")
        x_conv_2 = self.conv(x_emb, 128, activation=None, kernel_size=3, name="x_conv_2")
        x_conv = tf.concat([x_conv_1, x_conv_2], axis=-1)
        x_conv = tf.nn.relu(x_conv)
        x_conv = tf.nn.dropout(x_conv, self.dropout)

        x_conv = self.conv(x_conv, 256, activation=tf.nn.relu, kernel_size=5, name="x_conv_3")
        x_conv = tf.nn.dropout(x_conv, self.dropout)

        x_conv = self.conv(x_conv, 256, activation=tf.nn.relu, kernel_size=5, name="x_conv_4")
        x_conv = tf.nn.dropout(x_conv, self.dropout)

        x_conv = self.conv(x_conv, 256, activation=tf.nn.relu, kernel_size=5, name="x_conv_5")

        # BiLSTM
        # x_conv = self.add_bilstm_layer("bilstm", x_emb)

        # x_logit = tf.layers.dense(x_conv, num_classes)
        x_logit = self.add_full_connection_op("fc", x_conv, num_classes)

        self.labels_pred = tf.cast(tf.argmax(x_logit, axis=-1), tf.int32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x_logit, labels=self.input_y)
        mask = tf.sequence_mask(self.x_len, maxlen=83)
        loss = tf.boolean_mask(loss, mask)
        self.loss = tf.reduce_mean(loss)

        with tf.variable_scope("train_step"):
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.init = tf.global_variables_initializer()

    def conv(self, inputs, output_size, activation=None, kernel_size=1, name="conv", bias=True, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            # outputs = tf.layers.conv1d(inputs, output_size, kernel_size, padding='same', activation=activation)
            # return outputs
            shapes = inputs.shape.as_list()
            if len(shapes) > 4:
                raise NotImplementedError
            elif len(shapes) == 4:
                filter_shape = [1, kernel_size, shapes[-1], output_size]
                bias_shape = [1, 1, 1, output_size]
                strides = [1, 1, 1, 1]
            else:
                filter_shape = [kernel_size, shapes[-1], output_size]
                bias_shape = [1, 1, output_size]
                strides = 1
            conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d

            in_size = float(filter_shape[-2]) * float(kernel_size)
            stdv_ = 1. / math.sqrt(in_size)
            randomW_ = np.random.uniform(low=-stdv_, high=stdv_, size=filter_shape)
            randomb_ = np.random.uniform(low=-stdv_, high=stdv_, size=bias_shape)

            kernel_ = tf.Variable(randomW_, name="kernel_", dtype=tf.float32)
            outputs = conv_func(inputs, kernel_, strides, "SAME")
            if bias:
                outputs += tf.Variable(randomb_, name="bias_", dtype=tf.float32)
            if activation is not None:
                return activation(outputs)
            else:
                return outputs

    def add_full_connection_op(self, task_name, inputv, out_size, active_function=None):
        inputv_shape = inputv.get_shape().as_list()
        in_size = inputv_shape[-1]
        with tf.variable_scope("proj" + task_name):
            stdv_ = 1. / math.sqrt(in_size)
            randomW_ = np.random.uniform(low=-stdv_, high=stdv_, size=(in_size, out_size))
            randomb_ = np.random.uniform(low=-stdv_, high=stdv_, size=(out_size))
            W = tf.Variable(randomW_, name="W", dtype=tf.float32)
            b = tf.Variable(randomb_, name="b", dtype=tf.float32)

            ntime_steps = inputv_shape[1]
            output = tf.reshape(inputv, [-1, in_size])
            Wxb = tf.matmul(output, W) + b
            if active_function is not None:
                Wxb = active_function(Wxb)
            logits = tf.reshape(Wxb, [-1, ntime_steps, out_size])  # shape = (?,?,3)

        return logits

    def add_bilstm_layer(self, task_name, word_embeddings):
        with tf.variable_scope("bi-lstm" + task_name):
            lstm_cell_f = tf.contrib.rnn.LSTMCell(128)
            lstm_cell_b = tf.contrib.rnn.LSTMCell(128)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_f, lstm_cell_b, word_embeddings, sequence_length=self.x_len, dtype=tf.float32)

            output = tf.concat([output_fw, output_bw], axis=-1)  # shape = (?,?,600)
            word_embeddings_output = tf.nn.dropout(output, self.dropout)
        return word_embeddings_output

def valid_loss(sess, model, valid_X, valid_y, crf=False):
    losses = []
    for batch in batch_generator(valid_X, valid_y, crf=crf):
        batch_valid_X, batch_valid_y, batch_valid_X_len, batch_valid_X_mask = batch

        fd = {model.x: batch_valid_X, model.x_len: batch_valid_X_len, model.input_y: batch_valid_y, model.dropout: 1.0}
        loss = sess.run(model.loss, feed_dict=fd)
        losses.append(loss)

    return sum(losses) / len(losses)

def train(train_X, train_y, valid_X, valid_y, model, model_fn, epochs, lr, dropout, batch_size=128, crf=False):
    best_loss = float("inf")
    valid_history = []
    train_history = []
    saver = tf.train.Saver()

    gpuConfig = tf.ConfigProto()
    gpuConfig.gpu_options.allow_growth = True
    with tf.Session(config=gpuConfig) as sess:
        sess.run(model.init)

        for epoch in range(epochs):
            for batch in batch_generator(train_X, train_y, batch_size, crf=crf):
                batch_train_X, batch_train_y, batch_train_X_len, batch_train_X_mask = batch

                fd = {model.x: batch_train_X, model.x_len: batch_train_X_len, model.input_y: batch_train_y, model.dropout: dropout, model.lr: lr}
                _, train_loss, labels_pred = sess.run([model.train_op, model.loss, model.labels_pred], feed_dict=fd)

            print("epoch %d" % (epoch))
            loss = valid_loss(sess, model, train_X, train_y, crf=crf)
            print("train loss %f" % (loss))
            train_history.append(loss)
            loss = valid_loss(sess, model, valid_X, valid_y, crf=crf)
            valid_history.append(loss)
            if loss < best_loss:
                best_loss = loss
                saver.save(sess, model_fn)
            print("valid loss %f" % (loss), "valid best loss %f" % (best_loss))

            shuffle_idx = np.random.permutation(len(train_X))
            train_X = train_X[shuffle_idx]
            train_y = train_y[shuffle_idx]

    return train_history, valid_history


def run(domain, data_dir, model_dir, valid_split, runs, epochs, lr, dropout, batch_size=128):
    gen_emb = np.load(data_dir + "gen.vec.npy")
    # domain_emb = np.load(data_dir + "yelp_reviews_double.400d.txt.npy")
    domain_emb = np.load(data_dir + domain + "_emb.vec.npy")
    ae_data = np.load(data_dir + domain + ".npz")

    valid_X = ae_data['train_X'][-valid_split:]
    valid_y = ae_data['train_y'][-valid_split:]
    train_X = ae_data['train_X'][:-valid_split]
    train_y = ae_data['train_y'][:-valid_split]

    for r in range(runs):
        model = Model(gen_emb, domain_emb, 3, crf=False)
        train(train_X, train_y, valid_X, valid_y, model, model_dir + domain + '_tf_' + str(r), epochs, lr, dropout, batch_size, crf=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="model/")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--domain', type=str, default="laptop")
    parser.add_argument('--data_dir', type=str, default="data/prep_data/")
    parser.add_argument('--valid', type=int, default=150)  # number of validation data.
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.55)

    args = parser.parse_args()

    run(args.domain, args.data_dir, args.model_dir, args.valid, args.runs, args.epochs, args.lr, args.dropout, args.batch_size)
