"""
A demonstartion file showing how to generate TensorFlow TFRecord file.
The sample used here is the IBUG data set which is consist of 2 parts:
 1. a sample image.
 2. 68 facial landmarks.

Usage:
    python3 generate_tfrecord.py \
        --csv=data/ibug.csv \
        --image_dir=path_to_image \
        --mark_dir=path_to_marks \
        --output_file=data/ibug.record

"""
import json
import os
import sys

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

tf.enable_eager_execution()


# FLAGS, used as interface of user inputs.
flags = tf.app.flags
flags.DEFINE_string('csv', '', 'The csv file contains all file to be encoded.')
flags.DEFINE_string('image_dir', '', 'The path of images directory')
flags.DEFINE_string('mark_dir', '', 'The path of mark files directory')
flags.DEFINE_string('output_file', 'record.record',
                    'Where the record file should be placed.')
FLAGS = flags.FLAGS


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_feature_list(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_ibug_files(sample_name):
    """Generate sample files tuple"""
    image_file = os.path.join(FLAGS.image_dir, sample_name + '.jpg')
    mark_file = os.path.join(FLAGS.mark_dir, sample_name + '.json')
    return image_file, mark_file


def get_ibug_sample(image_file, mark_file):
    """Create a ibug sample from raw disk files."""
    # Keep the filename in the record for debug reasons.
    filename = image_file.split('/')[-1].split('.')[-2]
    ext_name = image_file.split('/')[-1].split('.')[-1]

    # Read encoded image file.
    with tf.gfile.GFile(image_file, 'rb') as fid:
        encoded_jpg = fid.read()

    # Read the marks from a json file.
    with open(mark_file) as fid:
        marks = json.load(fid)

    return {"filename": filename,
            "image_format": ext_name,
            "image": encoded_jpg,
            "marks": marks}


def create_tf_example(ibug_sample):
    """create TFRecord example from a data sample."""
    # Get required features ready.
    image_shape = tf.image.decode_jpeg(ibug_sample["image"]).shape

    # After getting all the features, time to generate a TensorFlow example.
    feature = {
        'image/height': _int64_feature(image_shape[0]),
        'image/width': _int64_feature(image_shape[1]),
        'image/depth': _int64_feature(image_shape[2]),
        'image/filename': _bytes_feature(ibug_sample['filename'].encode('utf8')),
        'image/encoded': _bytes_feature(ibug_sample['image']),
        'image/format': _bytes_feature(ibug_sample['image_format'].encode('utf8')),
        'label/marks': _float_feature_list(ibug_sample['marks'])
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

    return tf_example


def main(_):
    """Entrance"""
    tf_writer = tf.python_io.TFRecordWriter(FLAGS.output_file)

    samples = pd.read_csv(FLAGS.csv)
    for _, row in tqdm(samples.iterrows()):
        sample_name = row['file_basename']
        img_file, mark_file = get_ibug_files(sample_name)
        ibug_sample = get_ibug_sample(img_file, mark_file)
        tf_example = create_tf_example(ibug_sample)
        tf_writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    tf.app.run()
