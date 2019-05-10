# tfrecord_utility

Generate TFRecord file with TensorFlow.

## Getting Started
This is a collection of handy Python scripts related with TensorFlow TFRecord file generation.

- `split_data.ipynb`: A notebook shows how to split the full dataset into train, validation and test subsets.
- `generate_tfrecord.py`: Generate a TFRecord file.
- `view_record.py`: View the contents of a TFRecord file.

If you are looking for facial landmark points generation, please check out the 'ibug' branch.


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

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* The official TensorFlow data tutorial.



