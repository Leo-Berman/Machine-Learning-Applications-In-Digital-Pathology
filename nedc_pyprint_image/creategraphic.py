import pointwithin
import annotations
import matplotlib.pyplot as plt
from svstojpg import svs_to_jpg as stj
import svsdimensions as sd
def main():
    filepath = "/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a004_lvl000_t000.csv"
    imagepath = "/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a004_lvl000_t000.svs"
    HEADER,IDS,LABELS,COORDS = annotations.parse_annotations(filepath)
    
    
    w,h = sd.get_dimensions(imagepath)
    
    shapes = []
    coords = []

    # remove element z (depth)
    # reverse order of elements so it is (x,y)
    for x in COORDS:
        for y in x:
            y.pop()
            y.reverse()

    # generates border of the image
    x,y = pointwithin.get_border(pointwithin.generate_polygon(COORDS[0]))
    plt.plot(x,y)
    
    # generates polygon of regions within the image
    for i in range(len(COORDS)):
        shapes.append(pointwithin.generate_polygon(COORDS[i]))
        x,y = pointwithin.get_border(shapes[i])
        plt.plot(x,y)
        plt.text(COORDS[i][0][0],COORDS[i][0][1],LABELS[i])

    
    plt.xlim(0,w)
    plt.ylim(0,h)
    stj(imagepath,"./DATA/graphic")
    im = plt.imread("./DATA/graphic.jpg")
    plt.imshow(im,extent=[0,w,0,h])
    plt.savefig("./DATA/demo.png")

    return COORDS

main()