"""
Usage:
    python3 view_record.py --record=data.record
"""
import argparse

import numpy as np
import tensorflow as tf

import cv2

FLAGS = None


def _extract_feature(element):
    """
    Extract features from a single example from dataset.
    """
    features = tf.parse_single_example(
        element,
        # Defaults are not specified since both keys are required.
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'label/x': tf.FixedLenFeature([], tf.int64),
            'label/y': tf.FixedLenFeature([], tf.int64)
        })
    return features


def show_record(filenames):
    """.
    Show the TFRecord contents
    """
    # Generate dataset from TFRecord file.
    dataset = tf.data.TFRecordDataset(filenames)

    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()
    next_example = iterator.get_next()

    # Extract features from single example
    features = _extract_feature(next_example)
    image_decoded = tf.image.decode_image(features['image/encoded'])
    label_x = tf.cast(features['label/x'], tf.int32)
    label_y = tf.cast(features['label/y'], tf.int32)

    # Use openCV for preview
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    # Actrual session to run the graph.
    with tf.Session() as sess:
        while True:
            try:
                image_tensor, label_text = sess.run(
                    [image_decoded, (label_x, label_y)])

                # Use OpenCV to preview the image.
                image = np.array(image_tensor, np.uint8)
                cv2.imshow("image", image)
                cv2.waitKey(100)

                # Show the labels
                print(label_text)
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
