"""
Usage:
    python3 view_record.py --record=data.record
"""
import argparse

import cv2
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

FLAGS = None
IMG_SIZE = 128
MARK_SIZE = 68 * 2


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


def show_record(filenames):
    """Show the TFRecord contents"""
    # Generate dataset from TFRecord file.
    parsed_dataset = parse_tfrecord(filenames)

    for example in parsed_dataset:
        image_decoded = tf.image.decode_image(example['image/encoded']).numpy()
        height = example['image/height'].numpy()
        width = example['image/width'].numpy()
        depth = example['image/depth'].numpy()
        filename = example['image/filename'].numpy()
        img_format = example['image/format'].numpy()
        marks = example['label/marks'].numpy()

        print(filename, img_format, width, height, depth)
    
        # Use OpenCV to preview the image.
        image = np.array(image_decoded, np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the landmark on image
        landmark = np.reshape(marks, (-1, 2)) * IMG_SIZE
        _draw_landmark_point(image, landmark)

        # Show the result
        cv2.imshow("image", image)
        if cv2.waitKey() == 27:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        type=str,
        default="train.record",
        help="The record file."
    )
    args = parser.parse_args()
    show_record(args.record)
