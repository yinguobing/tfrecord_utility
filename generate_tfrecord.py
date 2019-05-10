"""
A demonstartion file showing how to generate TensorFlow TFRecord file.
The sample used here is a extended IBUG data set which is consist of three parts:
 1. a sample image.
 2. 68 facial landmarks.
 3. head pose: pitch, yaw and roll angels. (This is not in the original IBUG data)

Usage:
    python generate_tfrecord.py \
        --csv=data/ibug.csv \
        --img_dir=path_to_image \
        --mark_dir=path_to_marks \
        --pose_dir=path_to_pose \
        --output_file=ibug.record

"""
from __future__ import division, print_function

import argparse
import io
import json
import sys

import pandas as pd
import tensorflow as tf
from PIL import Image

FLAGS = None


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_ibug_sample(image_file, mark_file, pose_file):
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

    # Read the pose from a json file.
    with open(pose_file) as fid:
        pose = json.load(fid)

    return {"filename": filename,
            "image_format": ext_name,
            "image": encoded_jpg,
            "marks": marks,
            "pose": pose}


def _create_tf_example(ibug_sample):
    """create TFRecord example from a data sample."""
    # Get required features ready.
    image_shape = tf.image.decode_jpeg(ibug_sample["image"]).shape

    # After getting all the features, time to generate a TensorFlow example.
    feature = {
        'image/height': _int64_feature(image_shape[0]),
        'image/width': _int64_feature(image_shape[1]),
        'image/depth': _int64_feature(image_shape[2])
        'image/filename': _bytes_feature(ibug_sample['filename']),
        'image/encoded': _bytes_feature(ibug_sample['image']),
        'image/format': _bytes_feature(ibug_sample['image_format'),
        'label/marks': _float_feature(ibug_sample['marks']),
        'label/pose': _float_feature(ibug_sample['pose'])
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

    return tf_example


def main(unused_argv):
    """
    entrance
    """
    tf_writer = tf.python_io.TFRecordWriter(FLAGS.output_file)
    examples = pd.read_csv(FLAGS.csv_input)
    for _, row in examples.iterrows():
        current_example = _create_tf_example(row)
        tf_writer.write(current_example.SerializeToString())
    tf_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv_input',
        type=str,
        default='data/data.csv',
        help='Directory where the data file is.'
    )
    parser.add_argument(
        '--img_folder',
        type=str,
        default="images",
        help="Directory where the images live."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data.record",
        help="Where the record file should be placed."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
