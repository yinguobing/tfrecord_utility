# tfrecord_utility

Generate TFRecord file with TensorFlow.

## Getting Started
This is a collection of handy Python scripts related with TensorFlow TFRecord file generation.

- `split_data.ipynb`: A notebook shows how to split the full dataset into train, validation and test subsets.
- `generate_tfrecord.py`: Generate a TFRecord file.
- `view_record.py`: View the contents of a TFRecord file.


### Prerequisites

TensorFlow

```bash
pip3 install tensorflow
```

### Optional
- numpy
- pandas
- OpenCV (only if you need to run `view_record.py` to preview images).

### Installing

Git clone this repo then you are good to go.

```bash
git clone https://github.com/yinguobing/tfrecord_utility.git
```

## Running

### Generating IBUG TFRecord file.

Assuming you have IBUG data organized in the following manner:

- `/data/landmark/image` Extracted face images.
- `/data/landmark/mark` Extracted facial landmarks in JSON files.
- `/data/landmark/pose` Generated head pose in JSON files. Note this is not a part of the original IBUG data.

and you have list all the samples' name in a csv file:

`/data/landmark/ibug.csv`

and you want to put the generated TFRecord file here:

`/data/landmark/ibug.record`

Finally run the script like this:

```python
python3 generate_tfrecord.py --csv /data/landmark/ibug.csv --image_dir /data/landmark/image/ --mark_dir /data/landmark/mark/ --pose_dir /data/landmark/pose/ --output_file /data/landmark/ibug.record

```

The generated file `ibug.record` should be found.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* The official TensorFlow data tutorial.



