import nedc_mladp_fileio_tools as fileio_tools
import nedc_file_tools

def main():

    # set argument parsing
    #
    args_usage = "nedc_mladp_run.usage"
    args_help = "nedc_mladp_run.help"
    parameter_file = fileio_tools.parameters_only_args(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"gen_feats")
    output_path = parsed_parameters['output_file']

    
    
if __name__ == "__main__":
    main()
