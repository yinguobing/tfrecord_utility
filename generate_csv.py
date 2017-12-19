"""This script shows how to generate a file name list in CSV format"""
import csv
import os

FORMAT = ["json"]
FILE_DIR = "/home/robin/Documents/landmark/223K"


def main():
    """MAIN"""
    counter = 0
    field_names = ['jpg', 'json']
    with open('filelist.csv', 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for file_path, _, file_names in os.walk(FILE_DIR, followlinks=False):
            for file in file_names:
                if file.split(".")[-1] in FORMAT:
                    common_name = file.split(".")[-2]
                    jpg_file = str(os.path.join(file_path, common_name + '.jpg'))
                    json_file = str(os.path.join(file_path, common_name + '.json'))
                    writer.writeheader()
                    writer.writerow({'jpg': jpg_file, 'json': json_file})
                    counter += 1
                    # print(common_name)
    print("All done! {} files counted.".format(counter))

if __name__ == '__main__':
    main()
