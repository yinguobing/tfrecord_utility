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
IMG_SIZE = 24


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
        'label/marks': tf.FixedLenFeature([2], tf.float32),
        'label/pose': tf.FixedLenFeature([], tf.float32)
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.parse_single_example(example_proto, feature_description)

    parsed_dataset = dataset.map(_parse_function)
    return parsed_dataset


def _draw_landmark_point(image, points):
    """
    Draw landmark point on image.
    """
    for point in points:
        cv2.circle(image, (int(point[0]), int(
            point[1])), 2, (0, 255, 0), -1, cv2.LINE_AA)


def show_record(filenames):
    """.
    Show the TFRecord contents
    """
    # Generate dataset from TFRecord file.
    dataset = tf.data.TFRecordDataset(filenames)

    # Make dataset iterateable.
    iterator = dataset.make_one_shot_iterator()
    next_example = iterator.get_next()

    # Extract features from single example
    features = _extract_feature(next_example)
    image_decoded = tf.image.decode_image(features['image/encoded'])
    points = tf.cast(features['label/points'], tf.float32)

    # Use openCV for preview
    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    # Actual session to run the graph.
    with tf.Session() as sess:
        while True:
            try:
                image_tensor, raw_points = sess.run(
                    [image_decoded, points])

                # Use OpenCV to preview the image.
                image = np.array(image_tensor, np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw the landmark on image
                landmark = np.reshape(raw_points, (-1, 2)) * IMG_SIZE
                _draw_landmark_point(image, landmark)

                # Show the result
                cv2.imshow("image", image)
                if cv2.waitKey() == 27:
                    break

            except tf.errors.OutOfRangeError:
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
