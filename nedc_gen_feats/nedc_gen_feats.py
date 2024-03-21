import sys
sys.path.append("/data/isip/tools/linux_x64/nfc/class/python/nedc_sys_tools/nedc_file_tools")
import nedc_file_tools
import pandas
def parse_parameters():
    nedc_file_tools.load_parameters("param_format.txt")
    pandas.read_csv("parameters.csv")
def main():
    parse_parameters()
if __name__ == "__main__":
    main()
                
