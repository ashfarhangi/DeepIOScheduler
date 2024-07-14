import numpy as np
import os
import re

from model import SegSeq2SegSeq
from plot import plot_predictions
from data_utils import maybe_download, prepare_block_sequences, prepare_blocks_data, get_train_test_data, \
  rand_batch_gen, decode
# link and hyper parameterse
origin = 'http://skuld.cs.umass.edu/traces/storage/Financial1.spc.bz2'
vocab_freq = 5
win_size_ms = 64
segment_size = 1024
fname = 'Financial1.spc'
batch_size = 512
emb_dim = 128
epochs = 1
num_layers = 5
train_model = False
# Donwload by preprocess
maybe_download(datadir='datasets', fname=fname, url=origin)
_, block2idx, idx2block, _ = prepare_block_sequences(datadir="datasets", trace_file=fname, vocab_freq=vocab_freq, max_win_ms = win_size_ms, segment_size=segment_size)
sequences_filename = os.path.splitext(fname)[0]+".sequences"+str(vocab_freq)+"ss-"+str(segment_size)+"ws-"+str(win_size_ms)
train_input_path, train_label_path, test_input_path, test_label_path = prepare_blocks_data("datasets", sequences_filename, block2idx, train_test_split=0.1, tokenizer=None)
print(train_input_path, train_label_path, test_input_path, test_label_path)
(train_ip, train_lbl), (test_ip, test_lbl) = get_train_test_data(train_input_path, train_label_path, test_input_path, test_label_path)
print("train_ip size: {} train_lbl size: {} test_ip size: {}, test_lbl size:{}".format(train_ip.shape, train_lbl.shape, test_ip.shape, test_lbl.shape))
# ip[input] lbl[lable] 
ip_seq_len = train_ip.shape[-1]
lbl_seq_len = train_lbl.shape[-1]

ip_vocab_size = len(block2idx)
del block2idx
lbl_vocab_size = ip_vocab_size
model_name = os.path.splitext(fname)[0]+"segmentseq_model"

train_ip_count = train_ip.shape[0]
print("Train count: {}".format(train_ip_count))
print("Test count: {}".format(test_ip.shape[0]))

train_batch_gen = rand_batch_gen(train_ip, train_lbl, batch_size)
val_batch_gen = rand_batch_gen(test_ip, test_lbl, batch_size)
print("p1")

model = SegSeq2SegSeq(
  train_size = train_ip_count,
  batch_size = batch_size,
  xseq_len=ip_seq_len,
  yseq_len=lbl_seq_len,
  xvocab_size=ip_vocab_size,
  yvocab_size=lbl_vocab_size,
  ckpt_path='ckpt/',
  epochs=epochs,
  emb_dim=emb_dim,
  model_name=model_name,
  num_layers=num_layers)
print("p2")
sess = model.restore_last_session()
if train_model == True:
  sess = model.train(train_batch_gen, val_batch_gen)
else:
  input_, labels_ = val_batch_gen.__next__()
  output = model.predict(sess, input_)

  replies = []
  lbls = list()
  preds = list()

  for ii, il, oi in zip(input_.T, labels_.T, output):
    q = decode(sequence=ii, lookup=idx2block, separator=' ')
    l = decode(sequence=il, lookup=idx2block, separator=' ')
    o = decode(sequence=oi, lookup=idx2block, separator=' ')
    decoded = o.split(' ')

    if decoded.count('UNK') == 0:
      if decoded not in replies:
        if len(l) == len(o):
          print('i: [{0}]\na: [{1}]\np: [{2}]\n'.format(q, l, ' '.join(decoded)))
          print("{}".format("".join(["-" for i in range(80)])))
          lsplits = l.split()
          osplits = o.split()
          for lspl in lsplits:
            match = re.match(r"(\d+)(\w)", lspl)
            block, iotype = match.group(1), match.group(2)
            lbls.append(block)

          for osp in osplits:
            match = re.match(r"(\d+)(\w)", osp)
            block, iotype = match.group(1), match.group(2)
            preds.append(block)
        replies.append(decoded)

  preds = np.asarray(preds, dtype=np.int64)
  lbls = np.asarray(lbls, dtype=np.int64)
  plot_predictions(lbls, preds, segment_size, vocab_freq, win_size_ms)