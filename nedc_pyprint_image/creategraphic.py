import pointwithin
import annotations
import matplotlib.pyplot as plt

def main():
    filepath = "/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a001_lvl000_t000.csv"
    HEADER,IDS,LABELS,COORDS = annotations.parse_annotations(filepath)
    h = int(HEADER['height'])
    w = int(HEADER['width'])
    shapes = []
    coords = []
    for x in COORDS:
        for y in x:
            y.pop()
    x,y = pointwithin.get_border(pointwithin.generate_polygon(COORDS[0]))
    plt.plot(x,y)
    for i in range(len(COORDS)):
        shapes.append(pointwithin.generate_polygon(COORDS[i]))
        x,y = pointwithin.get_border(shapes[i])
        plt.plot(x,y)

    
    plt.xlim(0,w)
    plt.ylim(0,h)
    plt.savefig("./DATA/demo.png")


main()