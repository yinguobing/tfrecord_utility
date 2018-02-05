"""
Usage:
    # Create train record:
    python generate_tfrecord.py \
        --csv_input=data/data_train.csv \
        --img_folder=images \
        --output_file=train.record

    # Create validation record:
    python generate_tfrecord.py \
        --csv_input=data/data_validation.csv \
        --img_folder=images \
        --output_file=validation.record

    # Create test record:
    python generate_tfrecord.py \
        --csv_input=data/data_test.csv  \
        --img_folder=images \
        --output_file=test.record
"""
from __future__ import division, print_function

import argparse
import io
import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from generate_heat_map import put_heat

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _create_tf_example(data):
    """
    create TFRecord example from a single row of data.
    """
    # Encode jpg file.
    img_url = data['jpg']
    img_name = data['jpg'].split('.')[-2]

    # Read encoded image file, and get properties we need.
    with tf.gfile.GFile(img_url, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = img_name.encode('utf8')
    image_format = b'jpg'

    # Encode json file.
    json_url = data['json']
    with open(json_url) as json_file:
        points = json.load(json_file)

    # Get marks locations.
    label_marks = np.array(points, dtype=np.float32)
    label_marks = np.reshape(label_marks, (-1, 2)) * 64

    # Draw heat on map.
    heat_maps = []
    for point in label_marks:
        heat_map = np.zeros((64, 64), dtype=np.float32)
        put_heat(heat_map, point, sigma=1.9)
        heat_map_serialized = heat_map.flatten()
        heat_maps.append(heat_map_serialized)

    # Flatten heat maps as tf.train.feature accept only 1-d array.
    heat_maps = np.array(heat_maps, dtype=np.float32)
    heat_maps = heat_maps.reshape(-1,)

    # After getting all the features, time to generate a TensorFlow example.
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/filename': _bytes_feature(filename),
        'image/source_id': _bytes_feature(filename),
        'image/encoded': _bytes_feature(encoded_jpg),
        'image/format': _bytes_feature(image_format),
        'label/points': _float32_feature(points),
        'label/heat_maps': _float32_feature(heat_maps)
    }))
    return tf_example


def main(unused_argv):
    """
    entrance
    """
    counter = 0
    tf_writer = tf.python_io.TFRecordWriter(FLAGS.output_file)
    examples = pd.read_csv(FLAGS.csv_input)
    for _, row in examples.iterrows():
        counter += 1
        print(counter, row['jpg'])
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
