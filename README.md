Generate TFRecord file with TensorFlow.

If you are looking for facial landmark points generation, please check out the 'ibug' branch.

## Getting Started
- `split_data.ipynb`: A notebook shows how to split the full dataset into train, validation and test subsets.
- `generate_tfrecord.py`: Generate a TFRecord file.
- `view_record.py`: View the contents of a TFRecord file.

## Requirment
- TensorFlow 1.4
- numpy
- pandas
- OpenCV (only if you need to run `view_record.py` to preview images).
