"""
Usage:
    python3 view_record.py --record=data.record
"""
import argparse

import numpy as np
import tensorflow as tf

import cv2

FLAGS = None
IMG_SIZE = 24


def _extract_feature(element):
    """
    Extract features from a single example from dataset.
    """
    features = tf.parse_single_example(
        element,
        # Defaults are not specified since both keys are required.
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'label/points': tf.FixedLenFeature([2], tf.float32)
        })
    return features


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

    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()
    next_example = iterator.get_next()

    # Extract features from single example
    features = _extract_feature(next_example)
    image_decoded = tf.image.decode_image(features['image/encoded'])
    points = tf.cast(features['label/points'], tf.float32)

    # Use openCV for preview
    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    # Actrual session to run the graph.
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
