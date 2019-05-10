
"""Sample script of converting MXNET record into TensorFlow record."""
import mxnet as mx
import tensorflow as tf
from tqdm import tqdm

tf.enable_eager_execution()

# MXNET record:
INDEX = '/data/dataset/public/ms_celeb_1m/faces_emore/train.idx'
BIN = '/data/dataset/public/ms_celeb_1m/tfrecord/train.rec'

# The TFRecord file you want to generate.
TFRECORD = "/data/dataset/public/ms_celeb_1m/tfrecord/train-00009-of-00010"


# All raw values should be converted to a type compatible with tf.Example. Use
# the following functions to do these convertions.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_image_example(image_string, label):
    """Returns an tf.Example message with image encoded.
    Args:
        image_string: encoded image, NOT as numpy array.
        label: the label.
    Returns:
        a tf.Example.
    """
    image_shape = tf.image.decode_jpeg(image_string).shape

    # Create a dictionary with features that may be relevant.
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord(record_path):
    """Try to extract a image from the record file as jpg file."""
    raw_image_dataset = tf.data.TFRecordDataset(record_path)

    # Create a dictionary describing the features. This dict should be
    # consistent with the one used while generating the record file.
    image_feature_description = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.parse_single_example(example_proto, image_feature_description)

    parsed_dataset = raw_image_dataset.map(_parse_image_function)

    return parsed_dataset


def count_samples(parsed_dataset):
    counter = 0
    for image_features in parsed_dataset:
        counter += 1
    return counter


def save_one_sample_to_file(parsed_dataset, file_to_be_written='sample.jpg'):
    for image_features in parsed_dataset:
        label = image_features['label'].numpy()
        image_raw = image_features['image_raw']
        with tf.gfile.GFile(file_to_be_written, 'w') as fp:
            fp.write(image_raw.numpy())
        break
    print('One record parsed, label: {}'.format(label))
    print("An image extracted had been written to the current directory as {}".format(
        file_to_be_written))


def run():
    # Construct a MXNET record reader.
    print("Reading MXNET record...")
    mx_records = mx.recordio.MXIndexedRecordIO(INDEX, BIN, 'r')

    # Read the header to get total records count, which is embedded in the
    # first(0th) record.
    record_head = mx_records.read_idx(0)
    header, _ = mx.recordio.unpack(record_head)
    total_samples_num = int(header.label[0])
    print("Total records: {}".format(total_samples_num))

    # After getting the total count, we can loop through all of them and save
    # all examples in a TFRecord file.
    print("Converting record...")
    with tf.python_io.TFRecordWriter(TFRECORD) as writer:
        for i in tqdm(range(1, total_samples_num)):
            # Read a record from MXNET records with image_string and label.
            a_record = mx_records.read_idx(i)
            header, image_string = mx.recordio.unpack(a_record)
            label = int(header.label)

            # Convert the image and label to a tf.Example.
            tf_example = create_image_example(image_string, label)

            # Write the example to file.
            writer.write(tf_example.SerializeToString())

    print("All done. Record file is:\n{}".format(TFRECORD))


if __name__ == "__main__":
    # Generate TFRecord file.
    # run()

    # Test the file.
    # Parse the dataset.
    dataset = parse_tfrecord(TFRECORD)
    count_samples(dataset)

    # Extract one sample from the record file.
    save_one_sample_to_file(dataset, 'sample.jpg')
