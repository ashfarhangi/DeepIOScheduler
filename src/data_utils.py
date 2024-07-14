import bz2
import re
import os
import urllib
from pathlib import Path

from random import sample
from urllib.request import urlretrieve
from collections import Counter
import numpy as np
import time



from math import ceil

from scipy._lib.six import xrange
from tensorflow.python.platform import gfile

from plot import plot_frequencies

UNK = 0
counter = [['UNK', 0]]

def decompress_file(compressed, uncompressed):
  print('Uncompressing {} to {}'.format(compressed, uncompressed))
  with bz2.BZ2File(compressed, 'rb') as file:
    with open(uncompressed, 'wb') as new_file:
      for data in iter(lambda: file.read(100 * 1024), b''):
        new_file.write(data)

#downloads and decompresses the files.
def maybe_download(datadir, fname, url, untar=True):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(datadir):
    print("Creating directory %s" % datadir)
    os.mkdir(datadir)
  filepath_nobz2 = os.path.join(datadir, fname)
  filepath = filepath_nobz2 + '.bz2'
  if not os.path.exists(filepath_nobz2):
    if not os.path.exists(filepath):
      print("Downloading %s to %s" % (url, filepath))
      filepath, _ = urllib.request.urlretrieve(url, filepath)
      statinfo = os.stat(filepath)
      print("Successfully downloaded", fname, statinfo.st_size, "bytes")
      decompress_file(filepath, filepath_nobz2)
      return filepath_nobz2
    else:
      decompress_file(filepath, filepath_nobz2)
      return filepath_nobz2
  else:
    print("{} already exists".format(filepath_nobz2))
    return filepath_nobz2


def get_asus(fname):
  asus_set = set()
  with open(fname, 'r') as f:
    for line in f:
      splits = line.strip().strip(",")
      if splits[0] in asus_set:
        pass
      else:
        asus_set.add(splits[0])
  pass
  return len(asus_set), list(asus_set)

  def rand_batch_gen(x, y, batch_size):
    while True:
      sample_idx = sample(list(np.arange(len(x))), batch_size)
      yield x[sample_idx].T, y[sample_idx].T


def decode(sequence, lookup, separator=''):  # 0 used for padding, is ignored
  return separator.join([lookup[element] for element in sequence if element])


def iotype_to_rw(iotype):
  if iotype == 'Read':
    return 'R'
  else:
    return 'W'


def create_vocab_file(data_dir, trace_file, vocab_file, vocab_freq, segment_size=524288):
  vocab = {}
  blocks = list()
  blocks_iotype = list()

  with open(os.path.join(data_dir, trace_file), mode="r") as f:
    line_counter = 0
    for line in f:
      line_counter += 1
      if line.strip():
        splits = line.strip().split(',')
        offset = splits[1].strip()
        iotype = splits[3].strip()
        blocks_iotype.append(offset + iotype_to_rw(iotype))
        blocks.append(int(offset))
      if line_counter % 100000 == 0:
        print("  processing line %d" % line_counter)

  for block_iotype in blocks_iotype:
    if block_iotype in vocab:
      vocab[block_iotype] += 1
    else:
      vocab[block_iotype] = 1

  counter.extend([list(block_freq) for block_freq in Counter(vocab).most_common()
                  if block_freq[1] >= vocab_freq])

  vocab_list = [c[0] for c in counter]
  print("vocab list count: {}".format(len(vocab_list)))

  c_block_freq = dict()
  for c in counter:
    if c[0] != 'UNK':
      match = re.match(r"(\d+)(\w)", c[0])
      block, iotype = match.group(1), match.group(2)
      c_block_freq[int(block)] = int(c[1])
  plot_frequencies(freq_map=c_block_freq, max_freq=200, xlabel="Block", ylabel="Frequency",
                   title="Block Frequencies", fname="block-freqs.png")

  blocks_set = set(blocks)
  del blocks
  unique_blocks = list(blocks_set)
  unique_blocks.sort()
  lowest_offset = unique_blocks[1]
  highest_offset = unique_blocks[len(unique_blocks) - 1]
  print("lowest offset: {}, highest offset: {}, segment_size: {}".format(lowest_offset, highest_offset, segment_size))

  offset_ranges = OffsetRange(lowest_offset, highest_offset, segment_size)

  reduced_vocab_list = list()
  for w_idx, w in enumerate(vocab_list):
    if w == 'UNK':
      continue

    match = re.match(r"(\d+)(\w)", w)
    block, iotype = match.group(1), match.group(2)
    index = offset_ranges.find_range(int(block))
    offset = offset_ranges.ranges[index][0]
    reduced_vocab_list.append(str(offset) + iotype)

  reduced_vocab_set = set(reduced_vocab_list)
  reduced_vocab_list = list(reduced_vocab_set)

  vocabulary_path = os.path.join(data_dir, vocab_file)
  with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
    vocab_file.write("low:" + str(lowest_offset) + "\n")
    vocab_file.write("high:" + str(highest_offset) + "\n")
    vocab_file.write("segment_size:" + str(segment_size) + "\n")
    for w in reduced_vocab_list:
      vocab_file.write(w + "\n")

  print("Written vocabulary to: {}".format(vocabulary_path))
  print("Vocab size: {}".format(len(vocab_list)))
  print("Reduced Vocab size: {}".format(len(reduced_vocab_list)))
  return lowest_offset, highest_offset, segment_size, reduced_vocab_list


def prepare_blocks_vocab(data_dir, trace_file, vocab_file, vocab_freq, segment_size):
  """Create vocabulary file (if it does not exist yet) from trace file.

  Vocabulary contains the most-frequent tokens up to vocab_freq size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    data_dir: directory where the vocabulary file will be created/found.
    trace_file: data file that will be used to create vocabulary.
    vocab_file: file which contains known vocabulary for this dataset
    vocab_freq: # of times a block is required to be present in trace file
  """

  vocabulary_path = os.path.join(data_dir, vocab_file)
  vocabulary_posix_path = Path(vocabulary_path)
  vocabs = None
  if not vocabulary_posix_path.exists():
    print("Creating vocabulary file %s from data %s" % (vocab_file, trace_file))
    lowest_offset, highest_offset, segment_size, vocabs = create_vocab_file(data_dir, trace_file, vocab_file,
                                                                            vocab_freq, segment_size)
  else:
    print("{} already exists. Reusing ..".format(vocabulary_path))
    vocabs = list()
    lowest_offset = highest_offset = segment_size = -1

    with gfile.GFile(vocabulary_path, mode="r") as vocab_file:
      for line in vocab_file:
        if "low" in line:
          lowest_offset = line.strip().split(":")[1]
        elif "high" in line:
          highest_offset = line.strip().split(":")[1]
        elif "segment_size" in line:
          segment_size = line.strip().split(":")[1]
        else:
          vocabs.append(line.strip())

  block2idx = dict()
  for block in vocabs:
    block2idx[block] = len(block2idx)

  idx2block = dict(zip(block2idx.values(), block2idx.keys()))
  return lowest_offset, highest_offset, segment_size, vocabs, block2idx, idx2block


def _create_reduced_sequences(sequence, curr_max_win_ms, reduced_window_first_timestamp):
  reduced_window_sequences = list()
  reduced_window_sequence = " "
  splits = sequence.strip().split(" ")
  for offset_iotype_ts in splits:
    offset_iotype, ts = offset_iotype_ts.split(":")
    reduced_window_timestamp = ts

    if reduced_window_first_timestamp == -1:
      reduced_window_first_timestamp = reduced_window_timestamp
      reduced_window_sequence += " " + offset_iotype_ts
    else:
      # if int(reduced_window_timestamp) - int(reduced_window_first_timestamp) < curr_max_win_ms:
      if float(reduced_window_timestamp) - float(reduced_window_first_timestamp) < curr_max_win_ms:
        reduced_window_sequence += " " + offset_iotype_ts
      else:
        if reduced_window_sequence and len(reduced_window_sequence.strip().split()) > 1:
          reduced_window_sequences.append([reduced_window_sequence.strip()])
        # Start a new reduced window sequence
        reduced_window_sequence = " " + offset_iotype_ts
        reduced_window_first_timestamp = reduced_window_timestamp

  if reduced_window_sequence and len(reduced_window_sequence.strip().split()) > 1:
    reduced_window_sequences.append([reduced_window_sequence.strip()])

  return reduced_window_sequences


def get_optimal_number_steps(datadir, block_sequences_file):
  """
   Returns optimal number of steps for the given sequence of blocks in block_sequences_file
   If file does not exist, returns -1
  """
  block_counts = list()
  block_sequences_file_abspath = os.path.join(datadir, block_sequences_file)
  block_sequences_posix_file = Path(block_sequences_file_abspath)
  if block_sequences_posix_file.is_file():
    with open(block_sequences_file_abspath, 'r') as bsf:
      for line in bsf:
        if line:
          block_count = len(line.split(" "))
          block_counts.append(block_count)
    max_seq_len = max(block_counts)
    print("Max seq length: {} blocks ".format(max_seq_len))
    # Now bin the sequence lengths to figure out ideal sequence length to feed into the RNN
    bins = (np.linspace(1, max_seq_len, 20 + 1)).astype(np.int)
    binned = np.digitize(block_counts, bins)

    seq_length_frequencies = Counter(binned)
    plot_frequencies(freq_map=seq_length_frequencies, max_freq=2000,
                     xlabel="Sequence Length", ylabel="Frequency", title="Seq Length Frequencies",
                     fname="seq-length-freq.png")
    # plot_seq_length_frequencies(seq_length_frequencies)
    max_freq_seq_length_bin = max(seq_length_frequencies, key=lambda i: seq_length_frequencies[i])

    start_seq_length = -1
    end_seq_length = -1
    for i, bin in enumerate(bins):
      if max_freq_seq_length_bin == bin:
        start_seq_length = bin
        end_seq_length = bins[i + 1]

    print("bin: {} ie {} <= sequence lengths < {}".format(max_freq_seq_length_bin, start_seq_length, end_seq_length))
    print("Choosing # of times steps: {}".format(int(end_seq_length - 1 / 2)))

    return int(end_seq_length - 1 / 2)
  else:
    return -1


def get_split_by_asu_trace_files(datadir, trace_file):
  existing_asu_files = dict()
  with open(os.path.join(datadir, trace_file), 'r') as trace_file:
    line_count = 0
    for line in trace_file:
      if line.strip():
        if line_count and line_count % 100000 == 0:
          print("Done with line: {}".format(line_count))
        splits = line.strip().split(',')
        asu_number = splits[0].strip()
        offset = splits[1].strip()
        io_size = splits[2].strip()
        io_type = splits[3].strip()
        iotype = iotype_to_rw(io_type)
        timestamp = splits[4].strip()
        # timestamp = float(splits[4].strip()) * 1000  # convert secs to milliseconds
        asu_fname = "asu."+asu_number
        asu_fname_abspath = os.path.join(datadir, asu_fname)
        if asu_fname_abspath not in existing_asu_files:
          asu_fname_posix_path = Path(asu_fname_abspath)
          if not asu_fname_posix_path.exists():
            print("Creating: {}".format(asu_fname_abspath))
            fh = open(asu_fname_abspath, 'w')
            existing_asu_files[asu_fname_abspath] = fh
            print(str(timestamp) + "," + str(asu_number) + "," + str(io_size) + "," + io_type + "," + offset, file=fh)
        else:
          fh = existing_asu_files[asu_fname_abspath]
          print(str(timestamp) +", " + str(asu_number) + "," + str(io_size) + "," + io_type + "," + offset, file=fh)
      line_count += 1

  for f in existing_asu_files:
    fh = existing_asu_files[f]
    fh.close()
    print("Done writing to {}" + f)
  return existing_asu_files.keys()

def prepare_block_sequences(datadir="dataset", trace_file="Financial1.spc", vocab_freq=5, max_win_ms=1024, segment_size=524288):
  ''' Reads a trace file and creates sequences (sequences of block Ids) of blocks
  that were read or written less than max_win_ms apart. If a block is read or written
  after max_win milliseconds, a new sequence is created. Each block Id is appended with either
  an R (read) or W (write) to identify the type of operation.
  Eg sentences file:
  b1R:ts1 b2W:ts2 b3R:ts3 b4W:ts4
  b5W:ts21 b2R:ts22
  b6W:ts31 b9R:t39
  '''
  start_time = time.time()

  file_basename = os.path.splitext(trace_file)[0]
  vocab_file = file_basename + ".ss-"+str(segment_size)+".vocab-" + str(vocab_freq) + "ws-" + str(max_win_ms)
  print("trace file: {}".format(trace_file))
  lowest_offset, highest_offset, segment_size, blocks_vocab, block2idx, idx2block = prepare_blocks_vocab(datadir,
                                                                                                         trace_file,
                                                                                                         vocab_file,
                                                                                                         vocab_freq,
                                                                                                         segment_size)
  print("Vocab size: {}".format(len(block2idx)))
  sequence_count = 0

  block_sequences_filename = os.path.join(datadir, file_basename + ".sequences" + str(vocab_freq)) + "ss-" + str(segment_size) + "ws-" + str(max_win_ms)

  block_sequences_posix_file = Path(block_sequences_filename)
  if block_sequences_posix_file.exists():
    print("{} already exists. Reusing ..".format(block_sequences_filename))
    with open(block_sequences_filename, 'r') as bsf:
      for _ in bsf:
        sequence_count += 1
    return blocks_vocab, block2idx, idx2block, sequence_count
  else:
    # Perhaps a bzip compressed version of the file exists? Check for it
    bzipd_block_sequences_filename = block_sequences_filename + ".bz2"
    bzipd_block_sequences_posix_file = Path(bzipd_block_sequences_filename)
    if bzipd_block_sequences_posix_file.exists():
      decompress_file(bzipd_block_sequences_filename, block_sequences_filename)
      print("{} now exists. Reusing ..".format(block_sequences_filename))
      with open(block_sequences_filename, 'r') as bsf:
        for _ in bsf:
          sequence_count += 1
      return blocks_vocab, block2idx, idx2block, sequence_count

    print("{}  does not exist. Creating ..".format(block_sequences_filename))
    asu_file_count = 24
    file_not_found = False
    asu_file_paths = list()
    for i in range(asu_file_count):
      asu_file_path = Path(os.path.join(datadir, "asu."+str(i)))
      if not asu_file_path.exists():
        file_not_found = True
        break
      asu_file_paths.append(asu_file_path)

    if file_not_found:
      print("asu files don't exist. Creating .. ")
      asu_trace_files = get_split_by_asu_trace_files(datadir, trace_file)
    else:
      asu_trace_files = asu_file_paths

    offset_ranges = OffsetRange(lowest_offset, highest_offset, segment_size)


    with open(block_sequences_filename, 'w') as sequences_file:
      for file in asu_trace_files:
        with open(file.__str__(), 'r') as trace_file:
          prev_timestamp = -1
          sequence = ""
          for line_count, line in enumerate(trace_file):
            if line_count % 100000 == 0:
              if line_count > 0:
                print("Seen {} lines so far in {}".format(line_count, file.__str__()))

            if line.strip():
              splits = line.strip().split(',')
              timestamp = float(splits[0].strip()) * 1000 # convert secs to milliseconds
              io_type = splits[3].strip()
              iotype = iotype_to_rw(io_type)
              offset = splits[4].strip()
              index = offset_ranges.find_range(int(offset))
              ranged_offset = str(offset_ranges.ranges[index][0])

              # Ignore blocks that are not accessed min_freq number of times
              if ranged_offset + iotype not in block2idx:
                continue

              if prev_timestamp == -1:
                sequence += " " + ranged_offset + iotype + ":" + str(timestamp)
              else:
                if timestamp - prev_timestamp <= max_win_ms:
                  sequence += " " + ranged_offset + iotype + ":" + str(timestamp)
                else:  # timestamp difference is > max_win_ms. Commit sequence
                  if len(sequence.strip().split()) > 1:  # ignore single block accesses.
                    print(sequence, file=sequences_file)
                    sequence_count += 1
                  # Ignore block sequences that are far apart. In this case, no
                  # block was accessed until after max_win_ms. Ignore such single
                  # block accesses in max_win_ms durations
                  if sequence and len(sequence.split()) > 3:
                    curr_max_win_ms = max_win_ms

                    # Reduce max_win_ms by half and create block sequences. Doing so
                    # results in closely accessed blocks moving nearer in high
                    # dimensional space.
                    while curr_max_win_ms > 0:
                      curr_max_win_ms //= 2
                      reduced_win_first_ts = -1
                      reduced_window_sequences = _create_reduced_sequences(sequence,
                                                                           curr_max_win_ms,
                                                                           reduced_win_first_ts)
                      for reduced_seq in reduced_window_sequences:
                        if reduced_seq and len(reduced_seq) > 1:  # ignore single block accesses.
                          print("".join(reduced_seq), file=sequences_file)
                          sequence_count += 1

                  # Start new sequence
                  sequence = " " + ranged_offset + iotype + ":" + str(timestamp)
              prev_timestamp = timestamp

          if sequence and len(sequence.split()) > 1:  # ignore single block accesses.
            curr_max_win_ms = max_win_ms
            while curr_max_win_ms > 0:
              curr_max_win_ms //= 2
              reduced_win_first_ts = -1
              reduced_window_sequences = _create_reduced_sequences(sequence,
                                                                   curr_max_win_ms,
                                                                   reduced_win_first_ts)
              for reduced_seq in reduced_window_sequences:
                if reduced_seq and len(reduced_seq) > 1:  # ignore single block accesses.
                  print("".join(reduced_seq), file=sequences_file)
                  sequence_count += 1

            print(sequence, file=sequences_file)
            sequence_count += 1

      print("Sequences count: {}".format(sequence_count))
      end_time = time.time()
      print("Created {0} from {1} in {2:.2f} seconds".format(block_sequences_filename, trace_file,
                                                             end_time - start_time))
      return blocks_vocab, block2idx, idx2block, sequence_count


def _get_inputs_labels_from_sequence_line(num_steps, line, pad=True):
  inputs = list()
  labels = list()
  sequence = line.strip().split(" ")
  element_count = len(sequence)

  if element_count <= num_steps:
    return inputs, labels
  else:
    input_pos = 0
    while input_pos <= element_count - num_steps:
      ip_pos = 0
      ips = list()
      lbls = list()
      while ip_pos < num_steps:
        ips.append(sequence[input_pos + ip_pos].split(":")[0])
        try:
          lbls.append(sequence[input_pos + ip_pos + num_steps].split(":")[0])
        except IndexError:
          if pad:
            lbls.append(UNK)
          else:
            pass
        ip_pos += 1
      inputs.append(ips)
      labels.append(lbls)
      input_pos += 1
    return inputs, labels


def _data_to_token_ids(datas, target_path, vocab,
                       tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given block2idx map

  Args:
    datas: list of list of blocks
    target_path: path where the file with token-ids will be created.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  with gfile.GFile(target_path, mode="w") as tokens_file:
    ids = list()
    for data in datas:
      for dl in data:
        sentence_ids = list()
        for d in dl:
          try:
            id = vocab[d]
            sentence_ids.append(id)
          except KeyError:
            sentence_ids.append(UNK)
        if sentence_ids is not None:
          ids.append(sentence_ids)
    ids_np = np.asarray(ids)
    np.savez(target_path, ids_np)


'''
 generate batches, by random sampling a bunch of items
    yield (x_gen, y_gen)

'''


def rand_batch_gen(x, y, batch_size):
  while True:
    sample_idx = sample(list(np.arange(len(x))), batch_size)
    yield x[sample_idx].T, y[sample_idx].T


def prepare_blocks_data(datadir, block_sequences_file, block2idx,
                        train_test_split=0.1, tokenizer=None):
  """Get MSR traces into data_dir, create vocabularies and tokenize data.
Args:
    data_dir: directory in which the data sets will be stored.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 4 elements:
      (1) path to the token-ids for input blocks training data-set,
      (2) path to the token-ids for predicted blocks training data-set,
      (3) path to the token-ids for input blocks development data-set,
      (4) path to the token-ids for predicted blocks development data-set,

  """
  block_sequences_file_abspath = os.path.join(datadir, block_sequences_file)
  block_sequences_file_posixpath = Path(block_sequences_file_abspath)

  if not block_sequences_file_posixpath.exists():
    print("Block sequences file {} does not exist. Bailing ..".format(block_sequences_file_abspath))
    return
  else:
    num_steps = get_optimal_number_steps(datadir, block_sequences_file)
    with open(block_sequences_file_abspath, 'r') as f:
      inputss = list()
      labelss = list()
      for line in f:
        if len(line.strip().split(" ")) > num_steps:
          inputs, labels = _get_inputs_labels_from_sequence_line(num_steps,
                                                                 line=line,
                                                                 pad=True)
          if len(inputs) > 0:
            inputss.append(inputs)
          if len(labels) > 0:
            labelss.append(labels)

  total_count = len([inputs for inputs in inputss])
  test_count = int(total_count * train_test_split)

  train_inputs = list()
  train_labels = list()
  for i in range(total_count - test_count):
    train_inputs.append(inputss[i])
    train_labels.append(labelss[i])

  test_inputs = list()
  test_labels = list()
  for j in range(total_count - test_count, total_count):
    test_inputs.append(inputss[j])
    test_labels.append(labelss[j])

  del inputss
  del labelss

  train_input_ids_path = os.path.join(datadir, "train_input.ids.npz")
  train_label_ids_path = os.path.join(datadir, "train_label.ids.npz")

  _data_to_token_ids(train_inputs, train_input_ids_path, block2idx, tokenizer)
  del train_inputs
  _data_to_token_ids(train_labels, train_label_ids_path, block2idx, tokenizer)
  del train_labels

  test_input_ids_path = os.path.join(datadir, "test_input.ids.npz")
  test_label_ids_path = os.path.join(datadir, "test_label.ids.npz")

  _data_to_token_ids(test_inputs, test_input_ids_path, block2idx, tokenizer)
  del test_inputs
  _data_to_token_ids(test_labels, test_label_ids_path, block2idx, tokenizer)
  del test_labels

  return (train_input_ids_path, train_label_ids_path, test_input_ids_path, test_label_ids_path)


def get_train_test_data(train_input_path, train_label_path, test_input_path, test_label_path):
  tr_ip_npz = np.load(train_input_path)
  tr_lbl_npz = np.load(train_label_path)
  te_ip_npz = np.load(test_input_path)
  te_lbl_npz = np.load(test_label_path)
  return (tr_ip_npz[tr_ip_npz.files[0]], tr_lbl_npz[tr_lbl_npz.files[0]]), \
         (te_ip_npz[te_ip_npz.files[0]], te_lbl_npz[te_lbl_npz.files[0]])


def decode(sequence, lookup, separator=''):  # 0 used for padding, is ignored
  return separator.join([lookup[element] for element in sequence if element])


class OffsetRange(object):
  def __init__(self, low, high, block_size):
    block_size = int(block_size)
    low = int(low)
    high = int(high)
    self.block_size = block_size

    if low % 2 != 0:
      low -= low % block_size
    self.low = low
    self.high = high
    self.range_lows = list()
    self.range_highs = list()
    self.ranges = self._get_ranges()

  def _get_ranges(self):
    ranges = list()
    range_count = ceil(self.high / self.block_size)
    low = self.low
    high = low + self.block_size
    for _ in xrange(range_count):
      # ranges.append((low, low + self.block_size))
      self.range_lows.append(low)
      self.range_highs.append(high)
      # low = low + self.block_size + 1
      low = low + self.block_size
      high = low + self.block_size
    ranges = zip(self.range_lows, self.range_highs)
    return list(ranges)

  def find_range_in(self, offset, left, right):
    if right - left == 1:
      return right

    middle = int((left + right) / 2)

    if offset >= self.range_highs[0]:
      if offset >= self.range_highs[left] and offset < self.range_highs[right]:
        left = left
        right = middle
        return self.find_range_in(offset, left, right)
      else:
        left = right
        right = len(self.range_highs) - 1
        return self.find_range_in(offset, left, right)
    elif offset < self.range_highs[0]:
      if offset >= self.range_lows[left] and offset < self.range_highs[right]:
        left = left
        right = middle
        return self.find_range_in(offset, left, right)
      else:
        left = right
        right = len(self.range_lows) - 1
        return self.find_range_in(offset, left, right)

  def find_range(self, offset):
    if offset < self.range_lows[0] or offset > self.range_highs[len(self.range_highs) - 1]:
      return -1
    index = self.find_range_in(offset, 0, len(self.range_highs) - 1)
    if offset <= self.range_highs[0]:
      return index - 1
    if offset == self.range_highs[index] or offset == self.range_lows[index]:
      return index
    if offset >= self.range_lows[index] and offset <= self.range_highs[index]:
      return index
    if offset > self.range_highs[index]:
      return index + 1
    if offset < self.range_lows[index]:
      return index - 1
    else:
      return -1