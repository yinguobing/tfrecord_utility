"""Convert the content of the record.

Useage:

    python3 convert_record.py \
        --record input.record \
        --output output.record \
        --resize 128

"""
import argparse

import cv2
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument("--record", type=str,  default="train.record",
                    help="The record file.")
parser.add_argument("--output",    type=str,    default="output.record",
                    help="The output record file.")
parser.add_argument("--resize",    type=int,    default=128,
                    help="The new size of the images.")
args = parser.parse_args()


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


def parse_tfrecord(record_path):
    """Try to extract a image from the record file as jpg file."""
    dataset = tf.data.TFRecordDataset(record_path)

    # Create a dictionary describing the features. This dict should be
    # consistent with the one used while generating the record file.
    feature_description = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/depth': tf.FixedLenFeature([], tf.int64),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'label/marks': tf.FixedLenFeature([MARK_SIZE], tf.float32),
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.parse_single_example(example_proto, feature_description)

    parsed_dataset = dataset.map(_parse_function)
    return parsed_dataset


def _draw_landmark_point(image, points):
    """Draw landmark point on image."""
    for point in points:
        cv2.circle(image, (int(point[0]), int(
            point[1])), 2, (0, 255, 0), -1, cv2.LINE_AA)


def create_tf_example(ibug_sample):
    """create TFRecord example from a data sample."""
    # Get required features ready.
    image_shape = tf.image.decode_jpeg(ibug_sample["image"]).shape

    # After getting all the features, time to generate a TensorFlow example.
    feature = {
        'image/height': _int64_feature(image_shape[0]),
        'image/width': _int64_feature(image_shape[1]),
        'image/depth': _int64_feature(image_shape[2]),
        'image/filename': _bytes_feature(ibug_sample['filename']),
        'image/encoded': _bytes_feature(ibug_sample['image']),
        'image/format': _bytes_feature(ibug_sample['image_format']),
        'label/marks': _float_feature_list(ibug_sample['marks'])
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

    return tf_example


def convert_example(example):
    image_decoded = tf.image.decode_image(example['image/encoded'])
    image_resized = tf.image.resize(image_decoded, (args.resize, args.resize))
    image_resized = np.uint8(image_resized)
    image_encoded = tf.image.encode_jpeg(image_resized).numpy()

    filename = example['image/filename'].numpy()
    img_format = example['image/format'].numpy()
    marks = example['label/marks'].numpy()

    ibug_sample = {"filename": filename,
                   "image_format": img_format,
                   "image": image_encoded,
                   "marks": marks}

    return create_tf_example(ibug_sample)


def convert_dataset(input_record, output_record):
    # Generate dataset from TFRecord file.
    parsed_dataset = parse_tfrecord(input_record)

    tf_writer = tf.python_io.TFRecordWriter(output_record)

    for example in parsed_dataset:
        print('.', end=' ')
        new_example = convert_example(example)
        tf_writer.write(new_example.SerializeToString())


if __name__ == "__main__":
    convert_dataset(args.record, args.output)
