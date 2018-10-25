import tensorflow as tf
import numpy as np
import os
import datetime
import time
import codecs
import pickle
import json
import sys
from sklearn import metrics


class FlagAttr:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


def set_params_flags():
    flags = dict()

    flags["max_utter_length"] = 100
    flags["num_users"] = 4502

    flags["embedding_dim"] = 300
    flags["user_embedding_dim"] = 200
    flags["hidden_size"] = hidden_size
    flags["dropout_keep_prob"] = 0.5
    flags["l2_reg_lambda"] = 3.0

    flags["batch_size"] = 100
    flags["num_epochs"] = 200
    flags["display_every"] = 10
    flags["evaluate_every"] = 100
    flags["checkpoint_every"] = 100
    flags["num_checkpoints"] = 5
    flags["learning_rate"] = 1e-3

    flags["allow_soft_placement"] = True
    flags["log_device_placement"] = False

    return FlagAttr(flags)


class CDMM:
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 hidden_size, user_size, user_embedding_size, num_utters, l2_reg_lambda=0.0):

        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_user = tf.placeholder(tf.int32, shape=[None, num_utters], name='input_user')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.num_utters = num_utters
        self.utter_length = int(sequence_length / num_utters)
        self.hidden_size = hidden_size
        self.batch_size = tf.shape(self.input_text)[0]
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.embedding_size = embedding_size
        self.user_embedding_size = user_embedding_size

        self._instantiate_weights()
        l2_loss = tf.constant(0.0)

        input_text = tf.split(self.input_text, self.num_utters, axis=1)
        input_text = tf.stack(input_text, axis=1)
        input_user = self.input_user

        with tf.name_scope("text-embedding"):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_text")
            self.embedded_words = tf.nn.embedding_lookup(self.W_text, input_text)
            self.embedded_words_reshaped = tf.reshape(self.embedded_words,
                                                      shape=[-1, self.utter_length, embedding_size])
        with tf.name_scope("user-embedding"):
            self.W_user = tf.Variable(tf.random_uniform([user_size, user_embedding_size], -1.0, 1.0), name="W_user")
            self.embedded_users = tf.nn.embedding_lookup(self.W_user, input_user)

        with tf.name_scope("rnn"):
            input_text_reshaped = tf.reshape(input_text, shape=[-1, self.utter_length])
            relevant_input_text = tf.sign(tf.abs(input_text_reshaped))
            length_input_text = tf.cast(tf.reduce_sum(relevant_input_text, axis=1), tf.int32)
            reversed_embedded_words = tf.reverse_sequence(self.embedded_words_reshaped, length_input_text,
                                                          batch_dim=0, seq_dim=1)

            hidden_state_forward_list = self.gru_forward_word_level(self.embedded_words_reshaped, relevant_input_text)
            hidden_state_backward_list = self.gru_backward_word_level(reversed_embedded_words, relevant_input_text,
                                                                      length_input_text)

            self.hidden_state = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                                 zip(hidden_state_forward_list, hidden_state_backward_list)]

            utter_representation = self.hidden_state
            utter_representation, p_attention_word = self.attention_word_level(utter_representation)
            self.p_attention_word = tf.reshape(p_attention_word, shape=[-1, self.num_utters, self.utter_length])

            utter_representation = tf.reshape(utter_representation,
                                                 shape=[-1, self.num_utters, self.hidden_size*2])

            hidden_state_forward_utters = self.gru_forward_utter_level(utter_representation, self.embedded_users)
            hidden_state_backward_utters = self.gru_backward_utter_level(utter_representation, self.embedded_users)
            self.hidden_state_utter = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                                          zip(hidden_state_forward_utters, hidden_state_backward_utters)]

            conv_representation = self.hidden_state_utter
            conv_representation, p_attention_utter = self.attention_utter_level(conv_representation)
            self.p_attention_utter = tf.reshape(p_attention_utter, shape=[-1, self.num_utters])

            self.h_outputs = tf.nn.dropout(conv_representation, keep_prob=self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[hidden_size*4, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_outputs, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, axis=1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    def gru_forward_word_level(self, embedded_words, relevant_input_text):
        embedded_words_splitted = tf.split(embedded_words, self.utter_length, axis=1)
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]
        h_t = tf.ones((self.batch_size * self.num_utters, self.hidden_size))
        mask_helper_vec = tf.ones((self.batch_size * self.num_utters, 1), dtype=tf.int32)
        h_t_forward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):
            h_t_p1 = self.gru_single_step_word_level(Xt, h_t)
            mask_vec = tf.slice(relevant_input_text, begin=[0, time_step], size=[-1, 1])
            h_t = tf.add(tf.multiply(tf.cast(tf.subtract(mask_helper_vec, mask_vec), tf.float32), h_t),
                         tf.multiply(tf.cast(mask_vec, tf.float32), h_t_p1))
            h_t_forward_list.append(h_t)

        return h_t_forward_list

    def gru_forward_word_level_v2(self, embedded_words, relevant_input_text):
        embedded_words_splitted = tf.split(embedded_words, self.utter_length, axis=1)
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]

        h_t = tf.ones((self.batch_size * self.num_utters, self.hidden_size))
        h_t_forward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):
            h_t_p1 = self.gru_single_step_word_level(Xt, h_t)
            mask_vec = tf.slice(relevant_input_text, begin=[0, time_step], size=[-1, 1])
            h_t = tf.multiply(tf.cast(mask_vec, tf.float32), h_t_p1)
            h_t_forward_list.append(h_t)

        return h_t_forward_list

    def gru_backward_word_level(self, embedded_words, relevant_input_text, length_input_text):
        embedded_words_splitted = tf.split(embedded_words, self.utter_length, axis=1)
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]

        h_t = tf.ones((self.batch_size * self.num_utters, self.hidden_size))
        mask_helper_vec = tf.ones((self.batch_size * self.num_utters, 1), dtype=tf.int32)
        h_t_backward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):
            h_t_p1 = self.gru_single_step_word_level(Xt, h_t)
            mask_vec = tf.slice(relevant_input_text, begin=[0, time_step], size=[-1, 1])
            h_t = tf.add(tf.multiply(tf.cast(tf.subtract(mask_helper_vec, mask_vec), tf.float32), h_t),
                         tf.multiply(tf.cast(mask_vec, tf.float32), h_t_p1))
            h_t_backward_list.append(h_t)
        h_t_backward_tensor = tf.stack(h_t_backward_list, axis=1)
        h_t_backward_tensor = tf.reverse_sequence(h_t_backward_tensor, length_input_text, batch_dim=0, seq_dim=1)
        h_t_backward_list_splitted = tf.split(h_t_backward_tensor, self.utter_length, axis=1)
        h_t_backward_list = [tf.squeeze(x, axis=1) for x in h_t_backward_list_splitted]

        return h_t_backward_list

    def gru_backward_word_level_v2(self, embedded_words, relevant_input_text, length_input_text):
        embedded_words_splitted = tf.split(embedded_words, self.utter_length, axis=1)
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]

        h_t = tf.ones((self.batch_size * self.num_utters, self.hidden_size))
        h_t_backward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):
            h_t_p1 = self.gru_single_step_word_level(Xt, h_t)
            mask_vec = tf.slice(relevant_input_text, begin=[0, time_step], size=[-1, 1])
            h_t = tf.multiply(tf.cast(mask_vec, tf.float32), h_t_p1)
            h_t_backward_list.append(h_t)
        h_t_backward_tensor = tf.stack(h_t_backward_list, axis=1)
        h_t_backward_tensor = tf.reverse_sequence(h_t_backward_tensor, length_input_text, batch_dim=0, seq_dim=1)
        h_t_backward_list_splitted = tf.split(h_t_backward_tensor, self.utter_length, axis=1)
        h_t_backward_list = [tf.squeeze(x, axis=1) for x in h_t_backward_list_splitted]

        return h_t_backward_list

    def gru_forward_utter_level(self, utter_representation, embedded_users):
        utter_representation_splitted = tf.split(utter_representation, self.num_utters, axis=1)
        utter_representation_squeeze = [tf.squeeze(x, axis=1) for x in utter_representation_splitted]

        user_representation_splitted = tf.split(embedded_users, self.num_utters, axis=1)
        user_representation_squeeze = [tf.squeeze(x, axis=1) for x in user_representation_splitted]

        h_t = tf.ones((self.batch_size, self.hidden_size * 2))
        h_t_forward_list = []
        for Xt, user_t in zip(utter_representation_squeeze, user_representation_squeeze):
            h_t = self.gru_single_step_utter_level(Xt, h_t, user_t)
            h_t_forward_list.append(h_t)

        return h_t_forward_list

    def gru_backward_utter_level(self, utter_representation, embedded_users):
        utter_representation_splitted = tf.split(utter_representation, self.num_utters, axis=1)
        utter_representation_squeeze = [tf.squeeze(x, axis=1) for x in utter_representation_splitted]
        utter_representation_squeeze.reverse()

        user_representation_splitted = tf.split(embedded_users, self.num_utters, axis=1)
        user_representation_squeeze = [tf.squeeze(x, axis=1) for x in user_representation_splitted]
        user_representation_squeeze.reverse()

        h_t = tf.ones((self.batch_size, self.hidden_size * 2))
        h_t_forward_list = []
        for Xt, user_t in zip(utter_representation_squeeze, user_representation_squeeze):
            h_t = self.gru_single_step_utter_level(Xt, h_t, user_t)
            h_t_forward_list.append(h_t)
        h_t_forward_list.reverse()

        return h_t_forward_list

    def gru_single_step_word_level(self, Xt, h_t_minus_1):
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1, self.U_z) + self.b_z)
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1, self.U_r) + self.b_r)
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) + r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate

        return h_t

    def gru_single_step_utter_level(self, Xt, h_t_minus_1, user_t):
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z_utter) + tf.matmul(user_t, self.W_z_user)
                            + tf.matmul(h_t_minus_1, self.U_z_utter) + self.b_z_utter)
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r_utter) + tf.matmul(user_t, self.W_r_user)
                            + tf.matmul(h_t_minus_1, self.U_r_utter) + self.b_r_utter)
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h_utter) + tf.matmul(user_t, self.W_h_user)
                                  + r_t * (tf.matmul(h_t_minus_1, self.U_h_utter)) + self.b_h_utter)
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate

        return h_t

    def attention_word_level(self, hidden_state):
        hidden_state_ = tf.stack(hidden_state, axis=1)
        hidden_state_2 = tf.reshape(hidden_state_, shape=[-1, self.hidden_size * 2])
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2, self.W_w_attention_word) + self.W_b_attention_word)
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.utter_length, self.hidden_size * 2])

        hidden_state_context_similiarity = tf.multiply(hidden_representation, self.context_vecotor_word)
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity, axis=2)
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)

        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)

        p_attention_expanded = tf.expand_dims(p_attention, axis=2)

        utter_representation = tf.multiply(p_attention_expanded, hidden_state_)
        utter_representation = tf.reduce_sum(utter_representation, axis=1)

        return utter_representation, p_attention

    def attention_utter_level(self, hidden_state_utter):
        hidden_state_ = tf.stack(hidden_state_utter, axis=1)

        hidden_state_2 = tf.reshape(hidden_state_, shape=[-1, self.hidden_size * 4])
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     self.W_w_attention_utter) + self.W_b_attention_utter)
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.num_utters, self.hidden_size * 2])

        hidden_state_context_similiarity = tf.multiply(hidden_representation, self.context_vecotor_utter)
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity, axis=2)
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)

        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)

        p_attention_expanded = tf.expand_dims(p_attention, axis=2)
        utter_representation = tf.multiply(p_attention_expanded, hidden_state_)
        utter_representation = tf.reduce_sum(utter_representation, axis=1)

        return utter_representation, p_attention

    def _instantiate_weights(self):
        double_hidden_size = self.hidden_size * 2
        with tf.name_scope("gru_weights_word_level"):
            self.W_z = tf.get_variable("W_z", shape=[self.embedding_size, self.hidden_size], initializer=self.initializer)
            self.U_z = tf.get_variable("U_z", shape=[self.hidden_size, self.hidden_size], initializer=self.initializer)
            self.b_z = tf.get_variable("b_z", shape=[self.hidden_size])
            self.W_r = tf.get_variable("W_r", shape=[self.embedding_size, self.hidden_size], initializer=self.initializer)
            self.U_r = tf.get_variable("U_r", shape=[self.hidden_size, self.hidden_size], initializer=self.initializer)
            self.b_r = tf.get_variable("b_r", shape=[self.hidden_size])

            self.W_h = tf.get_variable("W_h", shape=[self.embedding_size, self.hidden_size], initializer=self.initializer)
            self.U_h = tf.get_variable("U_h", shape=[self.hidden_size, self.hidden_size], initializer=self.initializer)
            self.b_h = tf.get_variable("b_h", shape=[self.hidden_size])

        with tf.name_scope("gru_weights_utter_level"):
            self.W_z_utter = tf.get_variable("W_z_utter", shape=[double_hidden_size, double_hidden_size],
                                             initializer=self.initializer)
            self.U_z_utter = tf.get_variable("U_z_utter", shape=[double_hidden_size, double_hidden_size],
                                             initializer=self.initializer)
            self.b_z_utter = tf.get_variable("b_z_utter", shape=[double_hidden_size])
            self.W_r_utter = tf.get_variable("W_r_utter", shape=[double_hidden_size, double_hidden_size],
                                             initializer=self.initializer)
            self.U_r_utter = tf.get_variable("U_r_utter", shape=[double_hidden_size, double_hidden_size],
                                             initializer=self.initializer)
            self.b_r_utter = tf.get_variable("b_r_utter", shape=[double_hidden_size])

            self.W_h_utter = tf.get_variable("W_h_utter", shape=[double_hidden_size, double_hidden_size],
                                             initializer=self.initializer)
            self.U_h_utter = tf.get_variable("U_h_utter", shape=[double_hidden_size, double_hidden_size],
                                             initializer=self.initializer)
            self.b_h_utter = tf.get_variable("b_h_utter", shape=[double_hidden_size])

            self.W_z_user = tf.get_variable("W_z_user", shape=[self.user_embedding_size, double_hidden_size],
                                            initializer=self.initializer)
            self.W_r_user = tf.get_variable("W_r_user", shape=[self.user_embedding_size, double_hidden_size],
                                            initializer=self.initializer)
            self.W_h_user = tf.get_variable("W_h_user", shape=[self.user_embedding_size, double_hidden_size],
                                            initializer=self.initializer)

        with tf.name_scope("attention"):
            self.W_w_attention_word = tf.get_variable("W_w_attention_word",
                                                      shape=[double_hidden_size, double_hidden_size],
                                                      initializer=self.initializer)
            self.W_b_attention_word = tf.get_variable("W_b_attention_word", shape=[double_hidden_size])

            self.W_w_attention_utter = tf.get_variable("W_w_attention_utter",
                                                          shape=[self.hidden_size * 4, double_hidden_size],
                                                          initializer=self.initializer)
            self.W_b_attention_utter = tf.get_variable("W_b_attention_utter", shape=[double_hidden_size])
            self.context_vecotor_word = tf.get_variable("what_is_the_informative_word", shape=[double_hidden_size],
                                                        initializer=self.initializer)
            self.context_vecotor_utter = tf.get_variable("what_is_the_informative_utter",
                                                            shape=[double_hidden_size], initializer=self.initializer)


def load_data(conv_train_file_name, conv_val_file_name, embd_file_name):
    decision_train_file_name = "convs_train_decision.pkl"
    decision_val_file_name = "convs_val_decision.pkl"
    users_train_file_name = "users_kings_train.pkl"
    users_val_file_name = "users_kings_val.pkl"
    users_embd_file_name = "user_king_graph_emb.npy"

    with codecs.open(os.path.join(input_folder_root_path, conv_train_file_name), "rb") as pickle_f:
        train_convs = pickle.load(pickle_f)

    with codecs.open(os.path.join(input_folder_root_path, conv_val_file_name), "rb") as pickle_f:
        val_convs = pickle.load(pickle_f)

    with codecs.open(os.path.join(input_folder_root_path, decision_train_file_name), "rb") as pickle_f:
        train_decisions = pickle.load(pickle_f)

    with codecs.open(os.path.join(input_folder_root_path, decision_val_file_name), "rb") as pickle_f:
        val_decisions = pickle.load(pickle_f)

    with codecs.open(os.path.join(input_folder_root_path, users_train_file_name), "rb") as pickle_f:
        train_users = pickle.load(pickle_f)

    with codecs.open(os.path.join(input_folder_root_path, users_val_file_name), "rb") as pickle_f:
        val_users = pickle.load(pickle_f)

    with codecs.open(embd_file_name, "r", "utf-8") as json_f:
        vocab, embd = json.load(json_f)

    user_emb_mat = np.load(os.path.join(input_folder_root_path, users_embd_file_name))

    return train_convs, val_convs, train_decisions, val_decisions, vocab, embd, train_users, val_users, user_emb_mat


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def train():
    embd_file_name = input_folder_root_path + "filtered_cc_ko_300_JSP.json"
    conv_train_file_name = "convs_train.pkl"
    conv_val_file_name = "convs_val.pkl"

    train_convs, val_convs, train_decisions, val_decisions, vocab, embd, train_users, val_users, user_emb_mat = \
        load_data(conv_train_file_name, conv_val_file_name, embd_file_name)

    embedding = np.asarray(embd)
    embedding_dim = len(embd[0])
    vocab_size = len(vocab)
    num_convs, doc_length = train_convs.shape
    num_classes = train_decisions.shape[1]

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=flags.allow_soft_placement,
            log_device_placement=flags.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model_ins = CDMM(
                sequence_length=doc_length,
                num_classes=num_classes,
                vocab_size=vocab_size,
                embedding_size=embedding_dim,
                hidden_size=flags.hidden_size,
                l2_reg_lambda=flags.l2_reg_lambda,
                user_size=flags.num_users,
                user_embedding_size=flags.user_embedding_dim,
                num_utters=num_utters
            )

            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(flags.learning_rate).minimize(model_ins.loss, global_step=global_step)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            loss_summary = tf.summary.scalar("loss", model_ins.loss)
            acc_summary = tf.summary.scalar("accuracy", model_ins.accuracy)

            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=flags.num_checkpoints)

            sess.run(tf.global_variables_initializer())

            sess.run(model_ins.W_text.assign(embedding))
            print("Success to load pre-trained word2vec model!\n")

            sess.run(model_ins.W_user.assign(user_emb_mat))
            print("Success to load pre-trained user2vec model!\n")

            batches = batch_iter(list(zip(train_convs, train_users, train_decisions)),
                                 flags.batch_size, flags.num_epochs)
            for batch in batches:
                x_batch, users_batch, y_batch = zip(*batch)
                feed_dict = {
                    model_ins.input_text: x_batch,
                    model_ins.input_user: users_batch,
                    model_ins.input_y: y_batch,
                    model_ins.dropout_keep_prob: flags.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model_ins.loss, model_ins.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                if step % flags.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                if step % flags.evaluate_every == 0:
                    print("\nEvaluation:")
                    val_batches = batch_iter(list(zip(val_convs, val_users, val_decisions)), flags.batch_size, 1)
                    predictions = list()
                    accuracies = list()
                    labels = list()

                    for one_val_batch in val_batches:
                        x_val_batch, users_val_batch, y_val_batch = zip(*one_val_batch)
                        feed_dict_dev = {
                            model_ins.input_text: x_val_batch,
                            model_ins.input_user: users_val_batch,
                            model_ins.input_y: y_val_batch,
                            model_ins.dropout_keep_prob: 1.0
                        }
                        summaries_dev, loss, accuracy, predictions_part, p_attention_word_part, p_attention_utter_part \
                            = sess.run(
                            [dev_summary_op, model_ins.loss, model_ins.accuracy,
                             model_ins.predictions, model_ins.p_attention_word, model_ins.p_attention_utter],
                            feed_dict_dev)
                        dev_summary_writer.add_summary(summaries_dev, step)
                        predictions.append(predictions_part)
                        accuracies.append(accuracy)

                        y_val_batch_idx = np.argmax(np.asarray(y_val_batch), axis=1)
                        labels.append(y_val_batch_idx)

                    label_vec = np.concatenate(labels)
                    pred_vec = np.concatenate(predictions)
                    _, _, accuracy, _ = metrics.precision_recall_fscore_support(label_vec, pred_vec, average="weighted")

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, weighted F1 {:g}\n".format(time_str, step, loss, accuracy))

                if step % flags.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == "__main__":
    unk_id = 0
    eod_id = -1
    num_utters = 5
    top_k = 5
    top_k_utter = 2
    input_folder_root_path = "./inputs/"
    output_folder_root_path = "./outputs/"

    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    input_version = int(sys.argv[2])
    output_version = int(sys.argv[3])
    hidden_size = int(sys.argv[4])
    try:
        output_folder_root_path = sys.argv[5]
    except IndexError:
        print("Use output_folder_root_path as {}".format(output_folder_root_path))
        pass

    output_folder_path = output_folder_root_path + str(output_version) + "/"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    flags = set_params_flags()

    tf.app.run()
