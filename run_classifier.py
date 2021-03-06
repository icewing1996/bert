# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import itertools
import modeling
import optimization
import tokenization
import tensorflow as tf
import numpy as np

from lstm_crf_layer import MLP_and_softmax, BLSTM_CRF
from tensorflow.contrib.layers.python.layers import initializers
import tf_metrics
from collections import defaultdict
# from sklearn.metrics import classification_report

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", True, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("pred_batch_size", 8, "Total batch size for pred.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_ids, token_start_mask):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.token_start_mask = token_start_mask
    # self.bert_tokens = bert_tokens
    # self.token_start_idxs = token_start_idxs


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class CCGProcessor(DataProcessor):
  """Processor for the CNCCG data set."""

  def __init__(self, freq_cutoff=None, rank_cutoff=None):
    # freq_cutoff: only include labels that occurs at least this many times in combined dataset
    # rank_cutoff: only include the top N frequent labels in this dataset
    # Will choose the more limiting of the two criteria to return
    self.language = "zh"
    self.freq_cutoff = freq_cutoff
    self.rank_cutoff = rank_cutoff

  def get_train_examples(self, data_dir):
    lines = self._read_tsv(os.path.join(data_dir,'train.tsv'))
    examples = []
    for (i, line) in enumerate(lines):
      words = []
      labels = []
      for unit in line:
        split = unit.split('|')
        word = split[0]
        label = split[-1].strip()
        words.append(word)
        labels.append(label)

      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(' '.join(words))
      label = tokenization.convert_to_unicode(' '.join(labels))
      examples.append(
          InputExample(guid=guid, text_a=text_a, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
    true_labels = []
    examples = []
    for (i, line) in enumerate(lines):
      words = []
      labels = []
      for unit in line:
        split = unit.split('|')
        word = split[0]
        label = split[-1].strip()
        words.append(word)
        labels.append(label)

      guid = "dev-%d" % (i)
      text_a = tokenization.convert_to_unicode(' '.join(words))
      label = tokenization.convert_to_unicode(' '.join(labels))
      true_labels.append(labels)
      examples.append(
          InputExample(guid=guid, text_a=text_a, label=label))
    return examples, true_labels

  def get_test_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))
    true_labels = []
    examples = []
    for (i, line) in enumerate(lines):
      words = []
      labels = []
      for unit in line:
        split = unit.split('|')
        word = split[0]
        label = split[-1].strip()
        words.append(word)
        labels.append(label)

      guid = "test-%d" % (i)
      text_a = tokenization.convert_to_unicode(' '.join(words))
      label = tokenization.convert_to_unicode(' '.join(labels))
      true_labels.append(labels)
      examples.append(
          InputExample(guid=guid, text_a=text_a, label=label))
    return examples, true_labels

  def get_labels(self, data_dir):
    """See base class."""
    train_lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
    dev_lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
    test_lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))
    lines = train_lines + dev_lines + test_lines
    counts = defaultdict(int)
    for line in lines:
      for unit in line:
        label = tokenization.convert_to_unicode(unit.split('|')[-1])
        counts[label] += 1

    total_labels = len(counts)
    tf.logging.info("There are {} unique labels in total.".format(total_labels))


    labels = sorted(counts, key=counts.get, reverse=True)
    freq_cutoff = 1

    if self.rank_cutoff != None:
      freq_cutoff = max(counts[labels[self.rank_cutoff]], freq_cutoff)
      tf.logging.info("Rank cutoff corresponds to at least {} counts.".format(str(freq_cutoff)))

    if self.freq_cutoff != None:
      freq_cutoff = max(self.freq_cutoff, freq_cutoff)
      tf.logging.info("Frequency cutoff corresponds to at least {} counts.".format(str(self.freq_cutoff)))

    
    deleted_freq_count = 0
    deleted_label_count = 0
    for label, freq in counts.items():
      if freq < freq_cutoff:
        deleted_freq_count += freq
        deleted_label_count += 1
        del counts[label]

    fraction = deleted_freq_count / np.sum(counts.values())
    tf.logging.info("Ignoring {} of total data".format(str(fraction)))
    tf.logging.info("Keeping {} out of {} unique labels".format(str(total_labels-deleted_label_count), str(total_labels)))
    
    labels = sorted(counts, key=counts.get, reverse=True)
    return labels

    #lines = self._read_tsv(os.path.join(data_dir, "supertags.tsv"))
    #labels = []
    #for line in lines:
    #  labels.append(tokenization.convert_to_unicode(line[0]))
    #return labels

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = defaultdict(int)
  for (i, label) in enumerate(label_list):
    label_map[label] = i + 1

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2" 
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
  
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  label_ids = []
  orig_tokens = example.text_a.split()
  labels = example.label.split()
  bert_tokens = []
  # Token map will be an int -> int mapping between the `orig_tokens` index and the `bert_tokens` index.
  token_start_idxs = []
  token_start_mask = [0] * max_seq_length

  if len(label_ids) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  bert_tokens.append("[CLS]")
  label_ids.append(0)
  for orig_token, label in zip(orig_tokens, labels):
    sub_tokens = tokenizer.tokenize(orig_token)
    label_ids.extend([label_map[label]] * len(sub_tokens))
    # if label_map[label] != 0:
    token_start_idxs.append(len(bert_tokens))
    bert_tokens.extend(sub_tokens)

  if len(label_ids) > max_seq_length - 1:
      label_ids = label_ids[0:(max_seq_length - 1)]

  bert_tokens.append("[SEP]")
  label_ids.append(0)

  for start_idx in token_start_idxs:
    if start_idx >= max_seq_length: break
    token_start_mask[start_idx] = 1
  
  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # tf.logging.info("input_ids: %s" % str(len(input_ids)))
  # tf.logging.info("label_ids: %s" % str(len(label_ids)))
  # tf.logging.info("input_mask: %s" % str(len(input_mask)))
  # tf.logging.info("segment_ids: %s" % str(len(segment_ids)))

  # tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
  # tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
  # tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
  # tf.logging.info("labels: %s (id = %s)" % (example.label, " ".join([str(x) for x in label_ids])))

  assert len(input_ids) == len(label_ids) == len(input_mask) == len(segment_ids), 'input_ids: {}, label_ids: {}'.format(len(input_ids), len(label_ids))
  if len(input_ids) > max_seq_length:
      input_ids = input_ids[:max_seq_length]
      label_ids = label_ids[:max_seq_length]
      input_mask = input_mask[:max_seq_length]
      segment_ids = segment_ids[:max_seq_length]

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    label_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(label_ids) == max_seq_length
  
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("token_start_mask: %s" % " ".join([str(x) for x in token_start_mask]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("labels: %s (id = %s)" % (example.label, " ".join([str(x) for x in label_ids])))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_ids=label_ids,
      token_start_mask=token_start_mask)
      # bert_tokens=bert_tokens,
      # token_start_idxs=token_start_idxs
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_ids)
    features["token_start_mask"] = create_int_feature(feature.token_start_mask)
    # features["bert_tokens"] = create_int_feature(feature.bert_tokens)
    # features["token_start_idxs"] = create_int_feature(feature.token_start_idxs)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "token_start_mask": tf.FixedLenFeature([seq_length], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, token_start_mask,
                 hidden_size, num_layers, hidden_dropout_prob):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)
  embedding = model.get_sequence_output()
  batch_size, max_seq_length, embedding_size = modeling.get_shape_list(embedding, expected_rank=3)
  lengths = tf.reduce_sum(input_mask, reduction_indices=1)  # [batch_size] vector, sequence lengths of current batch
  mask = tf.to_float(token_start_mask)
  
  # mlp = MLP_and_softmax(embedded_chars=embedding, hidden_size=hidden_size, num_layers=num_layers,
  #                         hidden_dropout_prob=hidden_dropout_prob, initializers=initializers, num_labels=num_labels,
  #                         seq_length=max_seq_length, labels=labels, length_mask=mask, is_training=is_training)
  # rst = mlp.compute()
  blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=hidden_size, cell_type='lstm', num_layers=num_layers,
                          dropout_rate=1.0-hidden_dropout_prob, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
  rst = blstm_crf.add_blstm_crf_layer_not_really_working(crf_only=False)
  return rst

  # final_hidden = model.get_sequence_output()
  # final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  # batch_size = final_hidden_shape[0] 
  # seq_length = final_hidden_shape[1]
  # hidden_size = final_hidden_shape[2]

  # output_weights = tf.get_variable(
  #     "output_weights", [num_labels, hidden_size],
  #     initializer=tf.truncated_normal_initializer(stddev=0.02))

  # output_bias = tf.get_variable(
  #     "output_bias", [num_labels], initializer=tf.zeros_initializer())

  # with tf.variable_scope("loss"):
  #     input_mask = tf.to_float(input_mask)
  #     token_start_mask = tf.expand_dims(tf.to_float(token_start_mask), -1)

  #     final_hidden_matrix = tf.reshape(final_hidden, [batch_size * seq_length, hidden_size])
  #     logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b = True)
  #     logits = tf.nn.bias_add(logits, output_bias)

  #     logits = tf.reshape(logits, [batch_size, seq_length, num_labels])
  #     one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
  #     #loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits * token_start_mask, weights=input_mask, label_smoothing=0.1)
      
  #     log_probs = tf.nn.log_softmax(logits, axis=-1)

  #     _input_mask = tf.expand_dims(input_mask, -1)
  #     per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs * _input_mask * token_start_mask, axis=-1)
  #     loss = tf.reduce_mean(per_example_loss)
  #     logits = logits * _input_mask * token_start_mask
  
  # return (loss, per_example_loss, logits)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, hidden_size, num_layers, hidden_dropout_prob):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    token_start_mask = features["token_start_mask"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # (total_loss, per_example_loss, logits) = create_model(
    #     bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
    #     num_labels, use_one_hot_embeddings, token_start_mask)
    
    labels_one_hot = tf.one_hot(label_ids, num_labels)

    (total_loss, logits, pred_ids) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, labels_one_hot,
        num_labels, use_one_hot_embeddings, token_start_mask,
        hidden_size, num_layers, hidden_dropout_prob)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # tf.logging.info("**** Trainable Variables ****")
    # for var in tvars:
    #   init_string = ""
    #   if var.name in initialized_variable_names:
    #     init_string = ", *INIT_FROM_CKPT*"
    #   tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
    #                   init_string)

    
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      # def metric_fn(per_example_loss, label_ids, logits):
      def metric_fn(label_ids, logits, weight):
        #precision = tf_metrics.precision(label_ids, pred_ids, num_labels, weight)
        #recall = tf_metrics.recall(label_ids, pred_ids, num_labels, weight)
        #f = tf_metrics.f1(label_ids, pred_ids, num_labels, weight)
        accuracy = tf.metrics.accuracy(label_ids, pred_ids, weights=weight)

        # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        # mask = tf.greater(token_start_mask, 0)
        # token_start_mask_ = tf.to_float(token_start_mask)
        # label_ids = tf.boolean_mask(label_ids, mask)
        # predictions = tf.boolean_mask(predictions, mask)
        # accuracy = tf.metrics.accuracy(label_ids, predictions)
        # accuracy = tf.metrics.accuracy(label_ids, predictions, weights=token_start_mask_)
        # loss = tf.metrics.mean(per_example_loss)
        return {
            "eval_accuracy": accuracy,
            #"eval_precision": precision,
            #"eval_recall": recall,
            #"eval_f": f
      #     "eval_loss": loss,
        }

      weight = tf.to_float(token_start_mask)
      eval_metrics = (metric_fn, [label_ids, logits, weight])
      # eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      softmax = tf.nn.softmax(logits)
      log_probs = tf.math.log(softmax)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
         mode=mode,
         predictions={"log_probs": log_probs, "predictions": pred_ids, "input_ids": input_ids, "label_ids": label_ids, "token_start_mask": token_start_mask, "input_mask": input_mask},
         scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn

'''
# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn

# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features

def get_eval(pred_result, real_labels, label_list, max_seq_length):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    predictions = list(itertools.islice(pred_result, len(real_labels)))
    
    pred_labels = []
    real_labels_ = []

    for i in range(len(predictions)):
        real = real_labels[i]
        if len(real) > max_seq_length-1:
            continue
        real_ = [label_map[l] for l in real]
        real_labels_.extend(real_)
         
        pred = predictions[i]['values'][1 : len(real_)+1]
        pred_labels.extend(pred)
        assert len(real_) == len(pred)
    print(classification_report(real_labels_, pred_labels))
'''

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      'ccg':  CCGProcessor,
  }

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))
  
  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.pred_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples, real_labels = processor.get_dev_examples(FLAGS.data_dir)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
    
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      # Eval will be slightly WRONG on the TPU because it will truncate
      # the last batch.
      eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)
    
   # Eval code
    eval_result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(eval_result.keys()):
        tf.logging.info("  %s = %s", key, str(eval_result[key]))
        writer.write("%s = %s\n" % (key, str(eval_result[key])))
   
   # Metric code
   # pred_result = estimator.predict(input_fn=eval_input_fn)
   # get_eval(pred_result, real_labels, label_list, FLAGS.max_seq_length) 
   
  if FLAGS.do_predict:
    predict_examples, real_labels = processor.get_test_examples(FLAGS.data_dir)
    test_file = os.path.join(FLAGS.output_dir, "test.tf_record")
    file_based_convert_examples_to_features(
        predict_examples, label_list, FLAGS.max_seq_length, tokenizer, test_file)


    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False

    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.txt")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()