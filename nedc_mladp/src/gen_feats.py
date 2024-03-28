import sys
sys.path.append("../lib")
sys.path.append("/data/isip/tools/linux_x64/nfc/class/python/nedc_sys_tools")
import pandas
import nedc_regionid
import nedc_file_tools
import nedc_cmdl_parser


def read_file_lists(file_name):
    df = pandas.read_csv(file_name)
    return df.iloc[:, 0].to_list()

def main():
    args_usage = "gen_feats_usage.txt"
    args_help = "gen_feats_help.txt"
    argparser = nedc_cmdl_parser.Cmdl(args_usage,args_help)
    argparser.add_argument('-p', type = str)
    parsed_args = argparser.parse_args()
    print(print(parsed_args))
    # parsed_parameters = nedc_file_tools.load_parameters("picone_params_try.txt","gen_feats")
    # svs_list = read_file_lists(parsed_parameters['imagefile_list'])
    # csv_list = read_file_lists(parsed_parameters['labelfile_list'])
    # for svs,csv in zip(svs_list,csv_list):
    #     nedc_regionid.classify_frames(svs,csv,parsed_parameters['windowsize'],parsed_parameters['framesize'])


if __name__ == "__main__":
    main()