#all codes are garbage

import tensorflow as tf
import numpy as np
import sys
import time


class SegSeq2SegSeq(object):

    def __init__(self,
                 train_size,
                 batch_size,
                 xseq_len,
                 yseq_len,
                 xvocab_size,
                 yvocab_size,
                 emb_dim,
                 num_layers,
                 ckpt_path,
                 epochs,
                 model_name,
                 lr=0.0001):

        self.epochs = epochs
        self.batch_size = batch_size,
        self.train_size = train_size,
        self.xseq_len = xseq_len
        # self.pseq_len = pseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.model_name = model_name
        sys.stdout.write('Built Graph ..')
        # build comput graph
        setattr(tf.nn.rnn_cell.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
        setattr(tf.nn.rnn_cell.MultiRNNCell, '__deepcopy__', lambda self, _: self)

        # placeholders
        tf.reset_default_graph()

        #  encoder inputs : list of indices of length xseq_len
        self.enc_ip = [ tf.placeholder(shape=[None,], dtype=tf.int64,
                        name='ei_{}'.format(t)) for t in range(xseq_len) ]

        #  labels that represent the real outputs
        self.labels = [ tf.placeholder(shape=[None,], dtype=tf.int64,
                        name='ei_{}'.format(t)) for t in range(yseq_len) ]
        self.labels_fwd = [ tf.placeholder(shape=[None,], 
                            dtype=tf.int64, 
                            name='ei_{}'.format(t)) for t in range(yseq_len) ]

        #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
        # eif_
        # self.labels_bwd = [ tf.placeholder(shape=[None,], 
        #                     dtype=tf.int64, 
        #                     name='eib_{}'.format(t)) for t in range(pseq_len) ]
        self.dec_ip = [ tf.zeros_like(self.enc_ip[0], dtype=tf.int64,
                        name='GO') ] + self.labels[:-1]
        self.dec_ip_fwd = [ tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO') ] + self.labels_fwd[:-1]
        # self.dec_ip_bwd = [ tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO') ] + self.labels_bwd[:-1]
        # Basic LSTM cell wrapped in Dropout Wrapper
        self.keep_prob = tf.placeholder(tf.float32)

        # define the basic cell
        basic_cell = tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.BasicLSTMCell(emb_dim),
                    output_keep_prob=self.keep_prob)

        # stack cells together : n layered model
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True)


        # for parameter sharing between training model and testing model
        # with tf.variable_scope('decoder') as scope:
        #     # build the seq2seq model
        #     #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
        #     self.decode_outputs, self.decode_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        #         self.enc_ip,self.dec_ip, stacked_lstm, xvocab_size, yvocab_size, emb_dim)

        #     # share parameters
        #     scope.reuse_variables()

        #     # testing model, where output of previous timestep is fed as input to the next timestep
        #     self.decode_outputs_test, self.decode_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        #         self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size, emb_dim, feed_previous=True)
        with tf.variable_scope('decoder_fwd') as scope:
                # build the seq2seq model 
                #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimenstensorflow
       !!!!! FEED previous???
                self.decode_outputs_fwd, self.decode_states_fwd = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.enc_ip,self.dec_ip_fwd, stacked_lstm,
                                                    xvocab_size, yvocab_size, emb_dim)
                # share parameters
                scope.reuse_variables() 
                # testing model, where output of previous timestep is fed as input 
                #  to the next timestep
                self.decode_outputs_test_fwd, self.decode_states_test_fwd = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                    self.enc_ip, self.dec_ip_fwd, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
                    feed_previous=True)
        # with tf.variable_scope('decode_bwd') as scope:
        #         # build the seq2seq model 
        #         #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
        #         self.decode_outputs_bwd, self.decode_states_bwd = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.enc_ip,self.dec_ip_bwd, stacked_lstm,
        #                                             xvocab_size, yvocab_size, emb_dim)
        #         # share parameters
        #         scope.reuse_variables() 
        #         # testing model, where output of previous timestep is fed as input 
        #         #  to the next timestep
        #         self.decode_outputs_test_bwd, self.decode_states_test_bwd = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        #             self.enc_ip, self.dec_ip_bwd, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
        #             feed_previous=True)

            # now, for training,  build loss function

            # weighted loss
        loss_weights_fwd = [tf.ones_like(label, dtype=tf.float32) for label in self.labels_fwd]
        # loss_weights_bwd = [tf.ones_like(label, dtype=tf.float32) for label in self.labels_bwd]
        self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decode_outputs_fwd, self.labels_fwd, loss_weights_fwd, yvocab_size) 
        # \ + tf.contrib.legacy_seq2seq.sequence_loss(self.decode_outputs_bwd, self.labels_bwd, loss_weights_bwd, yvocab_size)
        tf.summary.scalar("loss", self.loss)
        print("e1")

            # train op to minimize the loss
        global_step = tf.Variable(0,name='global_step',trainable=False)
        self.train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=global_step, optimizer='Adam', learning_rate=lr, clip_gradients=2.)
        print("e2")
        sys.stdout.write('</log>')

    '''
        Training and Evaluation

    '''
    # get the feed dictionary
    def get_feed(self, X, Y, keep_prob):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels_fwd[t]: Y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = keep_prob # dropout prob
        # print("Feed dict: {}".format(feed_dict))
        return feed_dict

    # run one batch for training
    def train_batch(self, sess, train_batch_gen):
        # get batches
        batchX, batchY = train_batch_gen.__next__()
        # print("batchX shape: {} batchY shape: {}".format(batchX.shape, batchY.shape))
        # build feed
#!!!! Important play with the keep probality to see changes
        feed_dict = self.get_feed(batchX, batchY, keep_prob=0.5)
        _, loss_v = sess.run([self.train_op, self.loss], feed_dict)
        return loss_v

    def eval_step(self, sess, eval_batch_gen):
        # get batches
        batchX, batchY = eval_batch_gen.__next__()
        # print("batchX shape: {}".format(batchX.shape))
        # print("batchY shape: {}".format(batchY.shape))
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=1.)
        loss_v, dec_op_v_fwd = sess.run([self.loss, self.decode_outputs_test_fwd], feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices
        #  (interchange batch_size and timesteps dimensions
        # print("decoder output len before shape transform: {}".format(len(dec_op_v)))
        dec_op_v_fwd = np.array(dec_op_v_fwd).transpose([1,0,2])
        # print("dec_op_v shape: {}".format(dec_op_v.shape))
        return loss_v, dec_op_v_fwd, batchX, batchY

    # evaluate 'num_batches' batches
    def eval_batches(self, sess, eval_batch_gen, num_batches):
        losses = []
        for i in range(num_batches):
            # print("batch count: {}".format(i))
            # print("eval_batch_gen ip size: {}".format(eval_batch_gen.__next__()[0].shape))
            # print("eval_batch_gen lbl size: {}".format(eval_batch_gen.__next__()[1].shape))
            # print("num_batches: {}".format(num_batches))
            loss_v, dec_op_v_fwd, batchX, batchY = self.eval_step(sess, eval_batch_gen)
            losses.append(loss_v)
        return np.mean(losses)

    # finally the train function that
    #  runs the train_op in a session
    #   evaluates on valid set periodically
    #    prints statistics
    def train(self, train_set, valid_set, sess=None ):
        
        # we need to save the model periodically
        saver = tf.train.Saver()

        # if no session is given
        if not sess:
            # create a session
            sess = tf.Session()

            merged = tf.summary.merge_all()
            # summary_writer = tf.train.SummaryWriter('/tmp/umass-oltp', sess.graph)

            # init all variables
            sess.run(tf.global_variables_initializer())

        sys.stdout.write('\nTraining started ..\n')

        # run M epochs

        ip_size = self.train_size[0]
        b_size = self.batch_size[0]
        # print(ip_size)
        # print(b_size)
        num_steps = ip_size//b_size
        print(num_steps)
        prev_val_loss = 1000000
        prev_val_losses = list()
        patience = 100
        val_loss_eval_steps_count = 100
        for i in range(self.epochs):
            if i == 0:
                epoch_start_time = time.time()
            for j in range(num_steps):
                if j == 0:
                    steps_start_time = time.time()
                try:
                    pass
                    # summary, _ = self.train_batch(sess, merged, train_set)
                    # summary_writer.add_summary(summary)
                except KeyboardInterrupt: # this will most definitely happen, so handle it
                    print('Interrupted by user at iteration {}'.format(i))
                    self.session = sess
                if j and j % val_loss_eval_steps_count == 0:
                    val_loss = self.eval_batches(sess, valid_set, 8)
                    # writer.add_summary(summary)
                    steps_eval_time = time.time() - steps_start_time
                    print('val loss : {0:.6f} in {1:.2f} secs Epoch: {2}/{3} step/epoch'.format(val_loss,
                                                                                                steps_eval_time, j, i))
                    if i > 0 and val_loss > prev_val_loss:
                        saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                        prev_val_losses.append(val_loss)
                        if len(prev_val_losses) >= patience:
                            print("Early stopping at {0}/{1} with validation loss: {2}".format(j, i, val_loss))
                            return sess
                    if val_loss < prev_val_loss and prev_val_losses is not None:
                        prev_val_losses = list()
                    prev_val_loss = val_loss
                    steps_start_time = time.time()

            # save model to disk
            saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
            # evaluate to get validation loss
            val_loss = self.eval_batches(sess, valid_set, 8)  # TODO : and this
            epoch_eval_time = time.time() - epoch_start_time
            print('val loss : {0:.6f} in {1:.2f} secs Epoch: {2} epoch'.format(val_loss, epoch_eval_time, i))
            epoch_start_time = time.time()
            # print stats
            print('\nModel saved to disk at iteration #{} with loss:{}'.format(i, val_loss))
            sys.stdout.flush()
        return sess

    def restore_last_session(self):
        saver = tf.train.Saver()
        # create a session
        sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Restoring model from: {}".format(ckpt.model_checkpoint_path))
        else:
            print("No checkpoint to restore model from")
        # return to user
        return sess

    # prediction
    def predict(self, sess, X):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict[self.keep_prob] = 1.
        dec_op_v_fwd = sess.run(self.decode_outputs_test_fwd, feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices
        #  (interchange batch_size and timesteps dimensions
        dec_op_v_fwd = np.array(dec_op_v_fwd).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(dec_op_v_fwd, axis=2)