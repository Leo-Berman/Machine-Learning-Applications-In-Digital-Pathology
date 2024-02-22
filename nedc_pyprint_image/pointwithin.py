import shapely
import matplotlib.pyplot as plt

def generate_polygon(coords):
    shape = shapely.Polygon(coords)
    return shape

def get_border(shape):
    return (shape.exterior.xy)
def create_grid():
    square_coords = [(0,0),(0,10),(10,10),(10,0)]
    square = generate_polygon(square_coords)
    square_x,square_y = get_border(square)
    
    amorphous_coords=[(2,2),(3,4),(4,4),(5,5),(2,1)]
    amorphous = generate_polygon(amorphous_coords)
    amorphous_x,amorphous_y = get_border(amorphous)
    plt.plot(square_x,square_y)
    plt.plot(amorphous_x,amorphous_y)
    plt.savefig("DATA/demo.png")
