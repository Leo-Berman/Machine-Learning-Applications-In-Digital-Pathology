import sys
sys.path.append('/data/isip/tools/linux_x64/nfc/class/python/nedc_ann_dpath_tools')
import nedc_ann_dpath_tools as nadt

def parse_annotations(file):
    header,data = nadt.read(file)

    region_ids = []
    labels = []
    coords = []
    for i in data:
        region_ids.append(data[i]['region_id'])
        labels.append(data[i]['region_id'])
        coords.append(data[i]['coordinates'])

    return region_ids,labels,coords
def main():
    filepath = "/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a001_lvl000_t000.csv"
    IDS,LABELS,COORDS = parse_annotations(filepath)
    print(LABELS)
main()