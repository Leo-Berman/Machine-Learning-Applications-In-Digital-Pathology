import csv

# define default argument values
DEF_FRAMESIZE = int(-1)
DEF_WINDOWSIZE = int(-1)
# DEF_LVL = int(0)
# DEF_XOFF = float(0)
# DEF_YOFF = float(0)

# search for the specific parameter through the parameter file 
def parameter_search(parameter, parameter_file):
    # read the parameter file
    with open(parameter_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        # look for parameter
        for row in csv_reader:
            if row[0].startswith(parameter):
                # if there is no value associated with the parameter, set to default value
                if row[1] is None:
                    return default_value(parameter)
                # if there is a value associated with the parameter, return the value
                else:
                    return row[1]
            
def default_value(parameter):
    if parameter == "framesize":
        value = DEF_FRAMESIZE

    elif parameter == "windowsize":
        value = DEF_WINDOWSIZE
    
    # elif parameter == "level":
    #     value = DEF_LVL
    
    # elif parameter == "xoff":
    #     value = DEF_XOFF
    
    # elif parameter == "yoff":
    #     value = DEF_YOFF

def parse_param(parameter_file):
    param1 = "framesize"
    framesize = parameter_search(param1, parameter_file)

    param2 = "windowsize"
    windowsize = parameter_search(param2, parameter_file)

    # param3 = "level"
    # level = parameter_search(param3, parameter_file)

    # param4 = "xoff"
    # xoff = parameter_search(param4, parameter_file)

    # param5 = "yoff"
    # yoff = parameter_search(param5, parameter_file)

    param6 = "imagefile_list"
    imagefile_list = parameter_search(param6, parameter_file)

    param7 = "labelfile_list"
    labelfile_list = parameter_search(param7, parameter_file)

    return (imagefile_list, labelfile_list, int(windowsize), int(framesize))
