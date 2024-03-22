import sys
sys.path.append("../nedc_digipath_lib")
import nedc_digipath_lib



def main():
    svs_list_file, csv_list_file, window_size, frame_size = nedc_digipath_lib.fileio.parse_parameters()

    for svs,csv in zip(svs_list_file,csv_list_file):
        pass


if __name__ == "__main__":
    main()
                
