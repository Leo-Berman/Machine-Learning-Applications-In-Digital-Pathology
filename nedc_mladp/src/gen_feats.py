import sys
sys.path.append("../lib")

import pandas
import nedc_fileio
import nedc_regionid

def read_file_lists(file_name):
    df = pandas.read_csv(file_name)
    return df.iloc[:, 0].to_list()

def main():
    svs_list_file, csv_list_file, window_size, frame_size = nedc_fileio.parse_parameters("parameters.csv")
    svs_list = read_file_lists(svs_list_file)
    csv_list = read_file_lists(csv_list_file)
    for svs,csv in zip(svs_list,csv_list):
        nedc_regionid.classify_frames(svs,csv,window_size,frame_size)


if __name__ == "__main__":
    main()
                
