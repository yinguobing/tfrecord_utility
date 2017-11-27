# Create train data:
python3 generate_tfrecord.py \
    --csv_input=data/data_train.csv \
    --img_folder=images \
    --output_file=train.record

# Create validation data:
python3 generate_tfrecord.py \
    --csv_input=data/data_validation.csv \
    --img_folder=images \
    --output_file=validation.record

# Create test data:
python3 generate_tfrecord.py \
    --csv_input=data/data_test.csv  \
    --img_folder=images \
    --output_file=test.record
