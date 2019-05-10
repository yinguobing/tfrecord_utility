"""Shuffle samples in TFRecord files and reconstruct new record files.

Usage:
    python3 merge_records.py \
        --path_to_records=/path/to/records \
        --shuffle=true \
        --num_shards=10 \
        --output_file=record.record

"""
import logging
import os

import contextlib2
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util

logging.basicConfig(level=logging.DEBUG)

# FLAGS, used as interface of user inputs.
flags = tf.app.flags
flags.DEFINE_string('path_to_records', '', 'The path of record files')
flags.DEFINE_bool('shuffle', 'True', 'Shuffle records?')
flags.DEFINE_integer('num_shards', 10, 'Number of the shards')
flags.DEFINE_string('output_file', '', 'record.record')
FLAGS = flags.FLAGS


def list_records(target_path):
    file_list = []
    for file_path, _, current_files in os.walk(target_path, followlinks=False):
        for filename in current_files:
            # First make sure the file is exactly of the format we need.
            # Then process the file.
            file_url = os.path.join(file_path, filename)
            file_list.append(file_url)
    return file_list


def read_records(filenames, shuffle=True):
    """.
    Show the TFRecord contents
    """
    # Safety check, make sure the file exists.
    for file_name in filenames:
        assert os.path.exists(
            file_name), "File not found: {}".format(filenames)

    # Construct dataset from TFRecord file.
    dataset = tf.data.TFRecordDataset(filenames)

    # Shuffle dataset.
    dataset = dataset.shuffle(1024)

    # Make dataset iterateable.
    iterator = dataset.make_one_shot_iterator()
    next_example = iterator.get_next()

    return next_example


def main(_):
    record_files = list_records(FLAGS.path_to_records)
    logging.debug(
        "Number of records to be processed: {}".format(len(record_files)))
    next_example = read_records(record_files, shuffle=FLAGS.shuffle)

    # To maximize file I/O throughout, split the training data into pieces.
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_records = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, FLAGS.output_file, FLAGS.num_shards)

        with tf.Session() as sess:
            index = 0
            while True:
                try:
                    serialized_example = sess.run(next_example)
                    index += 1
                    output_shard_index = index % FLAGS.num_shards
                    output_records[output_shard_index].write(
                        serialized_example)

                    logging.debug("Samples processed: {}".format(index))

                except tf.errors.OutOfRangeError:
                    break


if __name__ == '__main__':
    tf.app.run()
