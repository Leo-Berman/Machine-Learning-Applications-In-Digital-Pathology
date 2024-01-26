import pandas as pd
import slideio
from matplotlib import pyplot as plt

def main():
    slide = slideio.open_slide("/data/isip/data/fccc_dpath/deidentified/v1.0.0/svs/00000/000000197/001003366/c50.2_c50.2/000000197_001003366_st065_xt1_t000.svs",'SVS')
    num_scenes = slide.num_scenes
    scene = slide.get_scene(0)
    print(num_scenes, scene.name, scene.rect, scene.num_channels)
    for channel in range(scene.num_channels):
        print(scene.get_channel_data_type(channel))

    image = scene.read_block(size=(500,0))
    plt.imshow(image)
    plt.savefig("demo.png")
main()