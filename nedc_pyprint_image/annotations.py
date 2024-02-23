# adding annotation reader file
import sys
sys.path.append('/data/isip/tools/linux_x64/nfc/class/python/nedc_ann_dpath_tools')
import nedc_ann_dpath_tools as nadt

# parse annotations
def parse_annotations(file):
    
    # read the data 
    header,data = nadt.read(file)
    
    # create lists for 
    region_ids = []
    labels = []
    coords = []
    
    # append to the lists
    for i in data:
        region_ids.append(data[i]['region_id'])
        labels.append(data[i]['text'])
        coords.append(data[i]['coordinates'])

    for i in range(len(coords)):
        for j in range(len(coords[i])):
            coords[i][j][1] = int(header['height'])-coords[i][j][1]
            coords[i][j].pop()
            coords[i][j].reverse()
    
    # return the lists
    return header,region_ids,labels,coords

# def main():
# filepath = "/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a001_lvl000_t000.csv"
# #     IDS,LABELS,COORDS = parse_annotations(filepath)
# # main()
# parse_annotations(filepath)